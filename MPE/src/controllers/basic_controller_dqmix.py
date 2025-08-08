from modules.models import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch
import torch.nn.functional as F 

# This multi-agent controller shares parameters between agents
class BasicMAC_DQMIX:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme) # for RL & online encoder (in online netwok)
        input_target_shape = self.args.obs_shape # for target network 
        input_transition_shape = self.n_agents * self.args.rnn_hidden_dim # for transion model in online network
        input_global_state_shape = self.args.state_shape # for global state encoder in online netwok

        # Initialized RNN agents 
        self._build_agents(input_shape) # for RL & online encoder (in online netwok)
        self._build_transition_model(input_transition_shape) # for transion model in online network
        self._build_target_agents(input_target_shape) # for target network 
        self._build_global_state_encoder(input_global_state_shape) # for global state encoder in online netwok
        # self._build_attention()

        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

        # For Self-Supervised Learning
        self.ssl_online_hidden_states = None
        self.ssl_momentum_hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = torch.nn.functional.softmax(agent_outs, dim=-1)

            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)

                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + torch.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    # For Self-Supervised Learning
    def online_forward(self, ep_batch, t):
        inputs = self._build_online_inputs(ep_batch, t)
        state_inputs = self._build_state_encoder_inputs(ep_batch, t)

        num_cur_step_total_ssl_online_hidden_states = []

        for input, state_input in zip(inputs, state_inputs):
            _, self.ssl_online_hidden_states = self.agent(input, self.ssl_online_hidden_states)

            if self.args.use_global: 
                self.state_embedding = self.global_state_encoder(state_input).unsqueeze(1).repeat(1, self.args.n_agents,1)

            if self.args.attention: 
                Q = self.ssl_online_hidden_states.reshape(32, self.n_agents, -1)
                K = self.ssl_online_hidden_states.reshape(32, self.n_agents, -1)
                V = self.ssl_online_hidden_states.reshape(32, self.n_agents, -1)

                d_k = K.shape[-1]

                attention_score = torch.bmm(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(d_k))

                attention_weights = F.softmax(attention_score, dim = 2)

                updated_obs = torch.bmm(attention_weights, V) 

                # element-wise multiplication
                if self.args.use_global: 
                    updated_obs = torch.mul(updated_obs, self.state_embedding)

                num_cur_step_total_ssl_online_hidden_states.append(updated_obs.view(self.args.batch_size * self.n_agents, -1))
                
            else: 
                updated_obs = self.ssl_online_hidden_states.reshape(32, self.n_agents, -1)
                
                if self.args.use_global: 
                    updated_obs = torch.mul(updated_obs, self.state_embedding)

                num_cur_step_total_ssl_online_hidden_states.append(updated_obs.view(self.args.batch_size * self.n_agents, -1))
        
        transtion_network_input = torch.stack(num_cur_step_total_ssl_online_hidden_states, dim=1)
        self.transition_model_hidden_states = self.tran_model(transtion_network_input, self.transition_model_hidden_states)

        return self.transition_model_hidden_states.view(self.args.batch_size, self.n_agents, -1)

    # @torch.no_grad()
    def momentum_forward(self, ep_batch, t):
        target_inputs = self._build_target_inputs(ep_batch, t)
        output = self.target_agent(target_inputs)

        return output

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

        if self.args.ssl_on:
            self.ssl_online_hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
            self.transition_model_hidden_states = self.tran_model.init_hidden()

    def parameters(self):

        return self.agent.parameters()

    def target_parameters(self):
        
        return self.target_agent.parameters()
    
    def transition_parameters(self):
        
        return self.tran_model.parameters()

    def global_state_encoder_parameters(self):
        
        return self.global_state_encoder.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.target_agent.cuda()
        self.tran_model.cuda()
        self.global_state_encoder.cuda()

    def save_models(self, path):
        torch.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(torch.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_transition_model(self, input_shape):
        self.tran_model = agent_REGISTRY["transition"](input_shape, self.args)

    # @torch.no_grad()
    def _build_target_agents(self, input_shape):
        self.target_agent = agent_REGISTRY[self.args.target_agent](input_shape, self.args.rnn_hidden_dim, self.args)

    def _build_global_state_encoder(self, input_shape):
        self.global_state_encoder = agent_REGISTRY[self.args.global_state_encoder](input_shape, self.args.rnn_hidden_dim, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av

        if self.args.obs_last_action:

            if t == 0:
                inputs.append(torch.zeros_like(batch["actions_onehot"][:, t]))

            else:
                inputs.append(batch["actions_onehot"][:, t-1])

        if self.args.obs_agent_id:
            inputs.append(torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = torch.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)

        return inputs
    
    def _build_online_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        input = [] 
        
        for idx in range(t, t+self.args.num_cur_step):
        
            inputs = []

            inputs.append(batch["obs"][:, idx])  # b1av

            if self.args.obs_last_action:

                inputs.append(batch["actions_onehot"][:, idx])

            if self.args.obs_agent_id:
                inputs.append(torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
            
            input.append(torch.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1))

        return input

    def _build_state_encoder_inputs(self, batch, t): 
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        input = [] 
        
        for idx in range(t, t+self.args.num_cur_step):
        
            inputs = []

            inputs.append(batch["state"][:, idx])  # b1av
            
            input.append(torch.cat([x.reshape(bs, -1) for x in inputs], dim=1))

        # input = torch.cat(input, dim=0)

        return input
    
    def _build_target_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t + self.args.num_cur_step + self.args.multi_step])  # b1av

        inputs = torch.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)

        return inputs


    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]

        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]

        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
    
