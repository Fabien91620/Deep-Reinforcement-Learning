use crate::do_not_touch::result_structures::PolicyAndActionValueFunction;
use crate::do_not_touch::secret_single_agent_env_wrapper::Env2;

/// Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
/// Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
/// Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
fn monte_carlo_es_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction {
    todo!()
}

/// Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
/// Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
/// Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction {
    todo!()
}

/// Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
/// Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
/// Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn off_policy_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction {
    todo!()
}

/// Creates a Secret Env2
/// Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
/// Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
fn monte_carlo_es_on_secret_env2() -> PolicyAndActionValueFunction {
    let _env = Env2::new();
    todo!()
}

/// Creates a Secret Env2
/// Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
/// Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn on_policy_first_visit_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction {
    let _env = Env2::new();
    todo!()
}

/// Creates a Secret Env2
/// Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
/// Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn off_policy_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction {
    let _env = Env2::new();
    todo!()
}


pub fn demo() {
    dbg!(monte_carlo_es_on_tic_tac_toe_solo());
    dbg!(on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo());
    dbg!(off_policy_monte_carlo_control_on_tic_tac_toe_solo());

    dbg!(monte_carlo_es_on_secret_env2());
    dbg!(on_policy_first_visit_monte_carlo_control_on_secret_env2());
    dbg!(off_policy_monte_carlo_control_on_secret_env2());
}