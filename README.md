## Deep Symbolic Optimization for Combinatorial Optimization: Accelerating Node Selection by Discovering Potential Heuristics

### How to Use
1. Generate benchmarks in `problem_generation`
2. Generate behaviors through `node_selection/behaviour_gen.py`
3. Train models using scripts in dir `learning`
4. Replace expressions in ` class OracleNodeSelectorEstimator_Symb` of `node_selection/node_selectors.py` with yours
