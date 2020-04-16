use numpy::{PyArray, PyArray1};
use petgraph::graph::{Graph, NodeIndex};
use petgraph::Directed;
use pyo3::prelude::*;
use std::collections::HashMap;
use tptp::parsers::{cnf_annotated, TPTPIterator};
use tptp::syntax::*;
use tptp::visitor::Visitor;

#[derive(Debug, Clone, Copy)]
pub enum NodeType {
    Variable,
    Functor,
    Argument,
    Application,
    Equality,
    Disequality,
    Negation,
    Axiom,
    NegatedConjecture,
    Selected,
    Action,
}

#[derive(Default)]
struct GraphBuilder {
    graph: Graph<NodeType, (), Directed, u32>,
    stack: Vec<Vec<NodeIndex>>,
    functors: HashMap<String, NodeIndex>,
    variables: HashMap<String, NodeIndex>,
    terms: HashMap<Vec<NodeIndex>, NodeIndex>,
}

impl GraphBuilder {
    fn children(&mut self) -> Vec<NodeIndex> {
        self.stack.pop().unwrap()
    }

    fn level(&mut self) {
        self.stack.push(vec![]);
    }

    fn last(&mut self) -> NodeIndex {
        self.stack.last_mut().unwrap().pop().unwrap()
    }

    fn record(&mut self, child: NodeIndex) {
        self.stack.last_mut().unwrap().push(child);
    }

    fn visit(&mut self, cnf: CnfAnnotated) -> NodeIndex {
        self.visit_cnf_annotated(cnf);
        self.last()
    }

    fn finish(mut self) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        for node_index in self.graph.node_indices() {
            self.graph.add_edge(node_index, node_index, ());
        }

        let nodes = self
            .graph
            .raw_nodes()
            .iter()
            .map(|n| n.weight as i64)
            .collect();
        let (sources, targets) = self
            .graph
            .raw_edges()
            .iter()
            .map(|e| (e.source().index() as i64, e.target().index() as i64))
            .unzip();

        (nodes, sources, targets)
    }
}

impl<'v> Visitor<'v> for GraphBuilder {
    fn visit_variable(&mut self, variable: Variable) {
        let key = format!("{}", variable);
        let variables = &mut self.variables;
        let graph = &mut self.graph;
        let index = *variables
            .entry(key)
            .or_insert_with(|| graph.add_node(NodeType::Variable));
        self.record(index)
    }

    fn visit_functor(&mut self, functor: Functor) {
        let key = format!("{}", functor);
        let functors = &mut self.functors;
        let graph = &mut self.graph;
        let index = *functors
            .entry(key)
            .or_insert_with(|| graph.add_node(NodeType::Functor));
        self.record(index)
    }

    fn visit_fof_plain_term(&mut self, term: FofPlainTerm) {
        use FofPlainTerm::*;
        match term {
            Constant(constant) => self.visit_constant(constant),
            Function(functor, arguments) => {
                self.level();
                self.visit_functor(functor);
                for argument in arguments.0 {
                    self.visit_fof_term(argument);
                }
                let key = self.children();

                let terms = &mut self.terms;
                let graph = &mut self.graph;
                let index = *terms.entry(key.clone()).or_insert_with(|| {
                    let mut arg_nodes = vec![];
                    for argument in &key[1..] {
                        let arg_node = graph.add_node(NodeType::Argument);
                        graph.add_edge(arg_node, *argument, ());
                        arg_nodes.push(arg_node);
                    }
                    let app_node = graph.add_node(NodeType::Application);
                    graph.add_edge(app_node, key[0], ());
                    for arg_node in &arg_nodes {
                        graph.add_edge(app_node, *arg_node, ());
                    }
                    for i in 1..arg_nodes.len() {
                        graph.add_edge(arg_nodes[i - 1], arg_nodes[i], ());
                    }
                    app_node
                });
                self.record(index)
            }
        }
    }

    fn visit_fof_defined_infix_formula(&mut self, defined_infix_formula: FofDefinedInfixFormula) {
        self.visit_fof_term(defined_infix_formula.left);
        self.visit_fof_term(defined_infix_formula.right);
        let right = self.last();
        let left = self.last();
        let equality = self.graph.add_node(NodeType::Equality);
        self.graph.add_edge(equality, left, ());
        self.graph.add_edge(equality, right, ());
        self.record(equality);
    }

    fn visit_literal(&mut self, literal: Literal) {
        use Literal::*;
        match literal {
            Atomic(a) => self.visit_fof_atomic_formula(a),
            NegatedAtomic(a) => {
                self.visit_fof_atomic_formula(a);
                let atomic = self.last();
                let negated = self.graph.add_node(NodeType::Negation);
                self.graph.add_edge(negated, atomic, ());
                self.record(negated);
            }
            Infix(infix) => {
                self.visit_fof_term(infix.left);
                self.visit_fof_term(infix.right);
                let right = self.last();
                let left = self.last();
                let disequality = self.graph.add_node(NodeType::Disequality);
                self.graph.add_edge(disequality, left, ());
                self.graph.add_edge(disequality, right, ());
                self.record(disequality);
            }
        }
    }

    fn visit_cnf_annotated(&mut self, annotated: CnfAnnotated) {
        self.variables.clear();
        self.level();
        self.visit_cnf_formula(annotated.formula);
        let clause = self
            .graph
            .add_node(if annotated.role == FormulaRole::NegatedConjecture {
                NodeType::NegatedConjecture
            } else {
                NodeType::Axiom
            });
        let children = self.children();
        for child in children {
            self.graph.add_edge(clause, child, ());
        }
        self.record(clause);
    }
}

#[pymodule]
fn clauses(_py: Python, module: &PyModule) -> PyResult<()> {
    type LongTensor = PyArray1<i64>;

    #[pyfn(module, "parse")]
    fn parse(bytes: &[u8]) -> PyResult<Vec<(bool, String)>> {
        let mut parsed = vec![];
        let mut parser = TPTPIterator::<()>::new(bytes);
        for input in &mut parser {
            let input = input.expect("parse error");
            if let TPTPInput::Annotated(input) = input {
                if let AnnotatedFormula::Cnf(cnf_annotated) = *input {
                    let is_conjecture = cnf_annotated.role == FormulaRole::NegatedConjecture;
                    parsed.push((is_conjecture, format!("{}", cnf_annotated.formula)));
                }
            }
        }
        Ok(parsed)
    }

    #[pyfn(module, "graph")]
    fn graph<'p>(
        py: Python<'p>,
        selected: Vec<&[u8]>,
        actions: Vec<&[u8]>,
    ) -> PyResult<(
        &'p LongTensor,
        &'p LongTensor,
        &'p LongTensor,
        &'p LongTensor,
    )> {
        let mut builder = GraphBuilder::default();
        builder.level();

        for clause in selected {
            let (_, clause) = cnf_annotated::<()>(clause).expect("parse error");
            let index = builder.visit(clause);
            let selected_node = builder.graph.add_node(NodeType::Selected);
            builder.graph.add_edge(selected_node, index, ());
        }

        let mut action_indices = vec![];
        for clause in actions {
            let (_, clause) = cnf_annotated::<()>(clause).expect("parse error");
            let index = builder.visit(clause);
            let action_node = builder.graph.add_node(NodeType::Action);
            builder.graph.add_edge(action_node, index, ());
            action_indices.push(action_node.index() as i64)
        }

        let (nodes, sources, targets) = builder.finish();
        let nodes = PyArray::from_vec(py, nodes);
        let sources = PyArray::from_vec(py, sources);
        let targets = PyArray::from_vec(py, targets);
        let action_indices = PyArray::from_vec(py, action_indices);
        Ok((nodes, sources, targets, action_indices))
    }

    Ok(())
}
