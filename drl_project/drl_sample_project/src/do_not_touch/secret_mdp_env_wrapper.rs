use drl_contracts::contracts::MDPEnv;

use crate::do_not_touch::bytes_wrapper::{free_wrapped_bytes, get_bytes, WrappedData};
use crate::do_not_touch::mdp_env_data_generated::{MDPEnvData, root_as_mdpenv_data};
use crate::do_not_touch::secret_envs_dynamic_libs_wrapper::SecretEnvDynamicLibWrapper;

struct SecretMdpEnv<'a> {
    env: *mut std::ffi::c_void,
    wrapper: SecretEnvDynamicLibWrapper,
    data_ptr: *mut WrappedData,
    data: MDPEnvData<'a>,
}

impl<'a> SecretMdpEnv<'a> {
    pub fn new(env: *mut std::ffi::c_void) -> Self {
        let wrapper = SecretEnvDynamicLibWrapper::new();
        unsafe {
            let data_ptr = wrapper.get_mdp_env_data()(env);
            let data_bytes = get_bytes(data_ptr);
            let data = root_as_mdpenv_data(data_bytes).unwrap();

            SecretMdpEnv {
                env,
                wrapper,
                data_ptr,
                data,
            }
        }
    }
}

impl<'a> Drop for SecretMdpEnv<'a> {
    fn drop(&mut self) {
        free_wrapped_bytes(self.data_ptr);
        unsafe {
            self.wrapper.delete_mdp_env()(self.env);
        }
    }
}

impl<'a> MDPEnv for SecretMdpEnv<'a> {
    fn states(&self) -> Vec<usize> {
        self.data.states().unwrap().iter().map(|v| v as usize).collect::<Vec<_>>()
    }

    fn actions(&self) -> Vec<usize> {
        self.data.actions().unwrap().iter().map(|v| v as usize).collect::<Vec<_>>()
    }

    fn rewards(&self) -> Vec<f32> {
        self.data.rewards().unwrap().iter().collect::<Vec<_>>()
    }

    fn is_state_terminal(&self, s: usize) -> bool {
        unsafe { self.wrapper.mdp_env_is_state_terminal()(self.env, s) }
    }

    fn transition_probability(&self, s: usize, a: usize, s_p: usize, r: f32) -> f32 {
        unsafe { self.wrapper.mdp_env_transition_probability()(self.env, s, a, s_p, r) }
    }

    fn view_state(&self, _s: usize) {
        println!("It's secret !")
    }
}

pub struct Env1<'a> {
    secret_env: SecretMdpEnv<'a>,
}

impl<'a> Env1<'a> {
    pub fn new() -> Self {
        unsafe {
            Env1 {
                secret_env: SecretMdpEnv::new(SecretEnvDynamicLibWrapper::new().create_secret_env1()())
            }
        }
    }
}

impl<'a> MDPEnv for Env1<'a> {
    fn states(&self) -> Vec<usize> {
        self.secret_env.states()
    }

    fn actions(&self) -> Vec<usize> {
        self.secret_env.actions()
    }

    fn rewards(&self) -> Vec<f32> {
        self.secret_env.rewards()
    }

    fn is_state_terminal(&self, s: usize) -> bool {
        self.secret_env.is_state_terminal(s)
    }

    fn transition_probability(&self, s: usize, a: usize, s_p: usize, r: f32) -> f32 {
        self.secret_env.transition_probability(s, a, s_p, r)
    }

    fn view_state(&self, s: usize) {
        self.secret_env.view_state(s)
    }
}