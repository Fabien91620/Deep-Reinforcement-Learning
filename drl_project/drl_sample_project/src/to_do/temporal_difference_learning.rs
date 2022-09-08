use crate::do_not_touch::result_structures::PolicyAndActionValueFunction;
use crate::do_not_touch::secret_single_agent_env_wrapper::Env3;

/// Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
/// Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
/// Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction {
    todo!()
}

/// Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
/// Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
/// Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn q_learning_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction {
    todo!()
}

/// Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
/// Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
/// Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn expected_sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction {
    todo!()
}

/// Creates a Secret Env3
/// Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
/// Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn sarsa_on_secret_env3() -> PolicyAndActionValueFunction {
    let _env = Env3::new();
    todo!()
}

/// Creates a Secret Env3
/// Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
/// Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn q_learning_on_secret_env3() -> PolicyAndActionValueFunction {
    let _env = Env3::new();
    todo!()
}

/// Creates a Secret Env3
/// Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
/// Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn expected_sarsa_on_secret_env3() -> PolicyAndActionValueFunction {
    let _env = Env3::new();
    todo!()
}


pub fn demo() {
    dbg!(sarsa_on_tic_tac_toe_solo());
    dbg!(q_learning_on_tic_tac_toe_solo());
    dbg!(expected_sarsa_on_tic_tac_toe_solo());

    dbg!(sarsa_on_secret_env3());
    dbg!(q_learning_on_secret_env3());
    dbg!(expected_sarsa_on_secret_env3());
}