use crate::to_do::{dynamic_programming, monte_carlo_methods, temporal_difference_learning};
use crate::do_not_touch::secret_mdp_env_wrapper::Env1;
use drl_contracts::contracts::{MDPEnv, SingleAgentEnv};
use crate::do_not_touch::secret_single_agent_env_wrapper::{Env2, Env3};

pub mod to_do;
pub mod do_not_touch;


fn main() {

    let env = Env1::new();
    dbg!(env.actions());
    dbg!(env.rewards());
    dbg!(env.states());
    dbg!(env.is_state_terminal(0));
    dbg!(env.transition_probability(0, 0, 0, 0f32));
    dbg!(env.view_state(0));

    let mut env = Env2::new();
    dbg!(env.reset());
    dbg!(env.available_actions_ids());
    dbg!(env.score());
    dbg!(env.act_with_action_id(0));
    dbg!(env.available_actions_ids());
    dbg!(env.state_id());
    dbg!(env.reset_random());

    let mut env = Env3::new();
    dbg!(env.reset());
    dbg!(env.available_actions_ids());
    dbg!(env.score());
    dbg!(env.act_with_action_id(0));
    dbg!(env.available_actions_ids());
    dbg!(env.state_id());
    dbg!(env.reset_random());

    dynamic_programming::demo();
    monte_carlo_methods::demo();
    temporal_difference_learning::demo();
}
