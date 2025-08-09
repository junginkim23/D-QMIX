import copy
import torch
import gc

from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import QMixer
from modules.heads.mlp import MLPHead
from utils.loss import MSFDMLoss


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.target_mac = copy.deepcopy(mac)
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None

        if args.mixer is not None:

            if args.mixer == "qmix":
                self.mixer = QMixer(args)

            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params = self.params + list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # For Self-Supervised Learning
        if self.args.name == 'd_qmix':
            self.ssl_loss = MSFDMLoss()

            # Define Momentum Mac
            self.momentum_mac = copy.deepcopy(mac)

            self.target_params = list(self.momentum_mac.target_parameters())

            # Define Online Projector & Predictor
            self.projector = MLPHead(
                in_features=args.rnn_hidden_dim,
                out_features=args.rnn_hidden_dim
            )
            self.predictor = MLPHead(
                in_features=args.rnn_hidden_dim,
                out_features=args.obs_shape
            )

            # Define Momentum Projector
            self.momentum_projector = MLPHead(
                in_features=args.rnn_hidden_dim,
                out_features=args.obs_shape
            )

            # Add Online Projector & Predictor Parameters
            self.params = self.params + list(self.projector.parameters())
            self.params = self.params + list(self.predictor.parameters())
            self.params = self.params + list(self.mac.transition_parameters())
            self.params = self.params + list(self.mac.global_state_encoder_parameters())

            self.target_params = self.target_params + list(self.momentum_projector.parameters())

            self.optimizer = torch.optim.RMSprop(
                params=self.params,
                lr=args.lr,
                alpha=args.optim_alpha,
                eps=args.optim_eps)
            
            self.target_optimizer = torch.optim.RMSprop(
                params=self.target_params,
                lr=args.lr,
                alpha=args.optim_alpha,
                eps=args.optim_eps)
            
        else:  # Only Multi-Agent Reinforcement Learning
            self.optimizer = torch.optim.RMSprop(
                params=self.params,
                lr=args.lr,
                alpha=args.optim_alpha,
                eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"] # number of actions 

        mac_out = []
        hidden_states = []
        self.mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            hidden_states.append(self.mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1))

        mac_out = torch.stack(mac_out, dim=1) 
        hidden_states = torch.stack(hidden_states, dim=1)

        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        target_mac_out = []
        target_hidden_states = []
        self.target_mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
            target_hidden_states.append(self.target_mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1))

        target_mac_out = torch.stack(target_mac_out[1:], dim=1)  # Concat across time
        target_hidden_states = torch.stack(target_hidden_states[1:], dim=1)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)

            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999

            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = torch.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            Q_total = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_Q_total = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        else:
            Q_total = chosen_action_qvals
            target_Q_total = target_max_qvals

        targets = rewards + self.args.gamma * (1 - terminated) * target_Q_total

        # Td-error
        td_error = (Q_total - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # For Self-Supervised Learning (Masked Reconstruction)
        if self.args.name == 'd_qmix':
            total_online_hidden_states = []
            self.mac.init_hidden(batch.batch_size)
            
            for t in range(0, batch.max_seq_length, self.args.num_cur_step):
                if t + self.args.multi_step + self.args.num_cur_step > (batch.max_seq_length - 1):
                    break
            
                else: 
                    online_hidden_states = self.mac.online_forward(batch, t=t)

                    total_online_hidden_states.append(
                        online_hidden_states) 

            total_online_hidden_states = torch.stack(total_online_hidden_states, dim=1) 

            # Momentum
            total_momentum_hidden_states = []
            self.momentum_mac.init_hidden(batch.batch_size)

            # with torch.no_grad():
            for t in range(0, batch.max_seq_length, self.args.num_cur_step):
                if t + self.args.multi_step + self.args.num_cur_step > (batch.max_seq_length - 1):
                    break

                momentum_hidden_states = self.momentum_mac.momentum_forward(batch, t=t)

                total_momentum_hidden_states.append(
                    momentum_hidden_states.view(batch.batch_size, self.args.n_agents, -1))

            total_momentum_hidden_states = torch.stack(total_momentum_hidden_states, dim=1)

            # Online projector & predictor
            projection = self.projector(total_online_hidden_states)
            prediction = self.predictor(projection)

            # Momentum projector
            # with torch.no_grad():
            momentum_projection = self.momentum_projector(total_momentum_hidden_states)
            ssl_loss = self.ssl_loss.calculate_loss(pred=prediction, true=momentum_projection)
            ssl_mean_loss = ssl_loss.mean()
            loss = loss + ssl_mean_loss

        # Optimise
        if self.args.name == 'd_qmix': 
            self.optimizer.zero_grad()
            self.target_optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            grad_norm_ = torch.nn.utils.clip_grad_norm_(self.target_params, self.args.grad_norm_clip)
            self.optimizer.step()
            self.target_optimizer.step()

        elif self.args.name in ['qmix']: 
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.optimizer.step()
        
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
            

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("target_grad_norm", grad_norm_, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (Q_total * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)

        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

        if self.args.ssl_on:
            self.momentum_mac.cuda()
            self.projector.cuda()
            self.predictor.cuda()
            self.momentum_projector.cuda()

        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)

        if self.mixer is not None:
            torch.save(self.mixer.state_dict(), "{}/mixer.th".format(path))

        torch.save(self.optimizer.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)

        if self.mixer is not None:
            self.mixer.load_state_dict(
                torch.load("{}/mixer.th".format(path),
                           map_location=lambda storage, loc: storage))

        self.optimizer.load_state_dict(
            torch.load("{}/opt.th".format(path),
                       map_location=lambda storage, loc: storage))
        
