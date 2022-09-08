use std::collections::HashMap;

/// Pi(s,a)
#[derive(Debug, Clone)]
pub struct Policy(HashMap<usize, HashMap<usize, f32>>);

/// V(s)
#[derive(Debug, Clone)]
pub struct ValueFunction(HashMap<usize, f32>);

/// Q(s,a)
#[derive(Debug, Clone)]
pub struct ActionValueFunction(HashMap<usize, HashMap<usize, f32>>);

/// Pi(s,a) and V(s)
#[derive(Debug, Clone)]
pub struct PolicyAndValueFunction {
    pi: Policy,
    v: ValueFunction,
}

/// Pi(s,a) and Q(s,a)
#[derive(Debug, Clone)]
pub struct PolicyAndActionValueFunction {
    pi: Policy,
    q: ActionValueFunction,
}