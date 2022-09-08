use libloading::{Library, Symbol};

use crate::do_not_touch::bytes_wrapper::WrappedData;
use crate::do_not_touch::SECRET_ENVS_DYN_LIB_PATHS;

#[allow(unused)]
type CreateMDPEnv = unsafe fn() -> *mut std::ffi::c_void;

type CreateSingleAgentEnv = unsafe fn() -> *mut std::ffi::c_void;
type ActOnSingleAgentEnv = unsafe fn(*mut std::ffi::c_void, usize);
type GetSingleAgentEnvStateData = unsafe fn(*mut std::ffi::c_void) -> *mut WrappedData;
type ResetSingleAgentEnv = unsafe fn(*mut std::ffi::c_void);
type ResetRandomSingleAgentEnv = unsafe fn(*mut std::ffi::c_void);
type DeleteSingleAgentEnv = unsafe fn(*mut std::ffi::c_void);
type GetMdpEnvData = unsafe fn(*mut std::ffi::c_void) -> *mut WrappedData;
type MdpEnvIsStateTerminal = unsafe fn(*mut std::ffi::c_void, s: usize) -> bool;
type MdpEnvTransitionProbability = unsafe fn(*mut std::ffi::c_void, s: usize, a: usize, s_p: usize, r: f32) -> f32;
type DeleteMdpEnv = unsafe fn(*mut std::ffi::c_void);

pub struct SecretEnvDynamicLibWrapper {
    lib: Library,
}

impl SecretEnvDynamicLibWrapper {
    pub fn new() -> Self {
        SecretEnvDynamicLibWrapper {
            lib: unsafe { Library::new(SECRET_ENVS_DYN_LIB_PATHS).unwrap() }
        }
    }

    pub fn create_secret_env1(&self) -> Symbol<CreateMDPEnv> {
        unsafe { self.lib.get(b"create_secret_env1").unwrap() }
    }

    pub fn create_secret_env2(&self) -> Symbol<CreateSingleAgentEnv> {
        unsafe { self.lib.get(b"create_secret_env2").unwrap() }
    }

    pub fn create_secret_env3(&self) -> Symbol<CreateSingleAgentEnv> {
        unsafe { self.lib.get(b"create_secret_env3").unwrap() }
    }

    pub fn act_on_single_agent_env(&self) -> Symbol<ActOnSingleAgentEnv> {
        unsafe { self.lib.get(b"act_on_single_agent_env").unwrap() }
    }

    pub fn get_single_agent_env_state_data(&self) -> Symbol<GetSingleAgentEnvStateData> {
        unsafe { self.lib.get(b"get_single_agent_env_state_data").unwrap() }
    }

    pub fn reset_single_agent_env(&self) -> Symbol<ResetSingleAgentEnv> {
        unsafe { self.lib.get(b"reset_single_agent_env").unwrap() }
    }

    pub fn reset_random_single_agent_env(&self) -> Symbol<ResetRandomSingleAgentEnv> {
        unsafe { self.lib.get(b"reset_random_single_agent_env").unwrap() }
    }

    pub fn delete_single_agent_env(&self) -> Symbol<DeleteSingleAgentEnv> {
        unsafe { self.lib.get(b"delete_single_agent_env").unwrap() }
    }

    pub fn get_mdp_env_data(&self) -> Symbol<GetMdpEnvData> {
        unsafe { self.lib.get(b"get_mdp_env_data").unwrap() }
    }

    pub fn mdp_env_is_state_terminal(&self) -> Symbol<MdpEnvIsStateTerminal> {
        unsafe { self.lib.get(b"mdp_env_is_state_terminal").unwrap() }
    }

    pub fn mdp_env_transition_probability(&self) -> Symbol<MdpEnvTransitionProbability> {
        unsafe { self.lib.get(b"mdp_env_transition_probability").unwrap() }
    }

    pub fn delete_mdp_env(&self) -> Symbol<DeleteMdpEnv> {
        unsafe { self.lib.get(b"delete_mdp_env").unwrap() }
    }
}