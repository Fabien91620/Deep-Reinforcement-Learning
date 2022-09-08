pub mod result_structures;
#[allow(unused)]
mod single_agent_env_state_data_generated;
#[allow(unused)]
mod mdp_env_data_generated;
mod bytes_wrapper;
mod secret_envs_dynamic_libs_wrapper;
pub mod secret_mdp_env_wrapper;
pub mod secret_single_agent_env_wrapper;

#[cfg(target_os = "windows")]
pub const SECRET_ENVS_DYN_LIB_PATHS: &str = &"secret_envs_dynamic_libs/drl_mystery_envs_wrapper.dll";

#[cfg(target_os = "linux")]
pub const SECRET_ENVS_DYN_LIB_PATHS: &str = &"secret_envs_dynamic_libs/libdrl_mystery_envs_wrapper.so";

#[cfg(target_os = "macos")]
pub const SECRET_ENVS_DYN_LIB_PATHS: &str = &"secret_envs_dynamic_libs/libdrl_mystery_envs_wrapper.dylib";
