use drl_contracts::contracts::SingleAgentEnv;

use crate::do_not_touch::bytes_wrapper::{free_wrapped_bytes, get_bytes, WrappedData};
use crate::do_not_touch::secret_envs_dynamic_libs_wrapper::SecretEnvDynamicLibWrapper;
use crate::do_not_touch::single_agent_env_state_data_generated::{root_as_single_agent_env_state_data, SingleAgentEnvStateData};

struct SecretSingleAgentEnv<'a> {
    env: *mut std::ffi::c_void,
    wrapper: SecretEnvDynamicLibWrapper,
    data_ptr: *mut WrappedData,
    data: SingleAgentEnvStateData<'a>,
}

impl<'a> SecretSingleAgentEnv<'a> {
    pub fn new(env: *mut std::ffi::c_void) -> Self {
        let wrapper = SecretEnvDynamicLibWrapper::new();
        unsafe {
            let data_ptr = wrapper.get_single_agent_env_state_data()(env);
            let data_bytes = get_bytes(data_ptr);
            let data = root_as_single_agent_env_state_data(data_bytes).unwrap();

            SecretSingleAgentEnv {
                env,
                wrapper,
                data_ptr,
                data,
            }
        }
    }
}

impl<'a> Drop for SecretSingleAgentEnv<'a> {
    fn drop(&mut self) {
        free_wrapped_bytes(self.data_ptr);
        unsafe {
            self.wrapper.delete_single_agent_env()(self.env);
        }
    }
}

impl<'a> SingleAgentEnv for SecretSingleAgentEnv<'a> {
    fn state_id(&self) -> usize {
        self.data.state_id() as usize
    }

    fn is_game_over(&self) -> bool {
        self.data.is_game_over()
    }

    fn act_with_action_id(&mut self, action_id: usize) {
        unsafe { self.wrapper.act_on_single_agent_env()(self.env, action_id) };
        free_wrapped_bytes(self.data_ptr);
        unsafe {
            self.data_ptr = self.wrapper.get_single_agent_env_state_data()(self.env);
            let data_bytes = get_bytes(self.data_ptr);
            self.data = root_as_single_agent_env_state_data(data_bytes).unwrap();
        }
    }

    fn score(&self) -> f32 {
        self.data.score()
    }

    fn available_actions_ids(&self) -> Vec<usize> {
        self.data.available_actions_ids().unwrap().iter().map(|a| a as usize).collect::<Vec<_>>()
    }

    fn reset(&mut self) {
        unsafe { self.wrapper.reset_single_agent_env()(self.env) }
        free_wrapped_bytes(self.data_ptr);
        unsafe {
            self.data_ptr = self.wrapper.get_single_agent_env_state_data()(self.env);
            let data_bytes = get_bytes(self.data_ptr);
            self.data = root_as_single_agent_env_state_data(data_bytes).unwrap();
        }
    }

    fn view(&self) {
        println!("It's secret !")
    }

    fn reset_random(&mut self) {
        unsafe { self.wrapper.reset_random_single_agent_env()(self.env) }
        free_wrapped_bytes(self.data_ptr);
        unsafe {
            self.data_ptr = self.wrapper.get_single_agent_env_state_data()(self.env);
            let data_bytes = get_bytes(self.data_ptr);
            self.data = root_as_single_agent_env_state_data(data_bytes).unwrap();
        }
    }
}

pub struct Env2<'a> {
    secret_env: SecretSingleAgentEnv<'a>,
}

impl<'a> Env2<'a> {
    pub fn new() -> Self {
        unsafe {
            Env2 {
                secret_env: SecretSingleAgentEnv::new(SecretEnvDynamicLibWrapper::new().create_secret_env2()())
            }
        }
    }
}

impl<'a> SingleAgentEnv for Env2<'a> {
    fn state_id(&self) -> usize {
        self.secret_env.state_id()
    }

    fn is_game_over(&self) -> bool {
        self.secret_env.is_game_over()
    }

    fn act_with_action_id(&mut self, action_id: usize) {
        self.secret_env.act_with_action_id(action_id)
    }

    fn score(&self) -> f32 {
        self.secret_env.score()
    }

    fn available_actions_ids(&self) -> Vec<usize> {
        self.secret_env.available_actions_ids()
    }

    fn reset(&mut self) {
        self.secret_env.reset()
    }

    fn view(&self) {
        self.secret_env.view()
    }

    fn reset_random(&mut self) {
        self.secret_env.reset_random()
    }
}

pub struct Env3<'a> {
    secret_env: SecretSingleAgentEnv<'a>,
}

impl<'a> Env3<'a> {
    pub fn new() -> Self {
        unsafe {
            Env3 {
                secret_env: SecretSingleAgentEnv::new(SecretEnvDynamicLibWrapper::new().create_secret_env3()())
            }
        }
    }
}

impl<'a> SingleAgentEnv for Env3<'a> {
    fn state_id(&self) -> usize {
        self.secret_env.state_id()
    }

    fn is_game_over(&self) -> bool {
        self.secret_env.is_game_over()
    }

    fn act_with_action_id(&mut self, action_id: usize) {
        self.secret_env.act_with_action_id(action_id)
    }

    fn score(&self) -> f32 {
        self.secret_env.score()
    }

    fn available_actions_ids(&self) -> Vec<usize> {
        self.secret_env.available_actions_ids()
    }

    fn reset(&mut self) {
        self.secret_env.reset()
    }

    fn view(&self) {
        self.secret_env.view();
    }

    fn reset_random(&mut self) {
        self.secret_env.reset_random()
    }
}