use crate::do_not_touch::result_structures::{ValueFunction, PolicyAndValueFunction};
use crate::do_not_touch::secret_mdp_env_wrapper::Env1;

/// Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
/// Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
/// Returns the Value function (V(s)) of this policy
fn policy_evaluation_on_line_world() -> ValueFunction {
    todo!()
}

/// Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
/// Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
/// Returns the Policy (Pi(s,a)) and its Value Function (V(s))
fn policy_iteration_on_line_world() -> PolicyAndValueFunction {
    todo!()
}

/// Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
/// Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
/// Returns the Policy (Pi(s,a)) and its Value Function (V(s))
fn value_iteration_on_line_world() -> PolicyAndValueFunction {
    todo!()
}

/// Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
/// Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
/// Returns the Value function (V(s)) of this policy
fn policy_evaluation_on_grid_world() -> ValueFunction {
    todo!()
}

/// Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
/// Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
/// Returns the Policy (Pi(s,a)) and its Value Function (V(s))
fn policy_iteration_on_grid_world() -> PolicyAndValueFunction {
    todo!()
}

/// Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
/// Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
/// Returns the Policy (Pi(s,a)) and its Value Function (V(s))
fn value_iteration_on_grid_world() -> PolicyAndValueFunction {
    todo!()
}

/// Creates a Secret Env1
/// Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
/// Returns the Value function (V(s)) of this policy
fn policy_evaluation_on_secret_env1() -> ValueFunction {
    let _env = Env1::new();
    todo!()
}

/// Creates a Secret Env1
/// Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
/// Returns the Policy (Pi(s,a)) and its Value Function (V(s))
fn policy_iteration_on_secret_env1() -> PolicyAndValueFunction {
    let _env = Env1::new();
    todo!()
}

/// Creates a Secret Env1
/// Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
/// Prints the Policy (Pi(s,a)) and its Value Function (V(s))
fn value_iteration_on_secret_env1() -> PolicyAndValueFunction {
    let _env = Env1::new();
    todo!()
}

pub fn demo() {
    dbg!(policy_evaluation_on_line_world());
    dbg!(policy_iteration_on_line_world());
    dbg!(value_iteration_on_line_world());

    dbg!(policy_evaluation_on_grid_world());
    dbg!(policy_iteration_on_grid_world());
    dbg!(value_iteration_on_grid_world());

    dbg!(policy_evaluation_on_secret_env1());
    dbg!(policy_iteration_on_secret_env1());
    dbg!(value_iteration_on_secret_env1());
}