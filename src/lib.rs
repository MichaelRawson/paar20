use numpy::{PyArray, PyArray1};
use petgraph::graph::{Graph, NodeIndex};
use petgraph::Directed;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use std::collections::HashMap;
use tptp::parsers::{cnf_annotated, tptp_input_iterator};
use tptp::syntax::*;

#[repr(u8)]
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
    Action,
    Selected,
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

    fn visit(&mut self, annotated: CnfAnnotated, selected: bool) {
        self.visit_cnf_annotated(annotated);
        let clause = self.last();
        let node = self.graph.add_node(if selected {
            NodeType::Selected
        } else {
            NodeType::Action
        });
        self.graph.add_edge(node, clause, ());
        self.record(node);
    }

    fn finish(mut self) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
        let nodes = self.graph
            .raw_nodes()
            .into_iter()
            .map(|n| n.weight as i32)
            .collect();

        let (sources, targets) = self.graph
            .raw_edges()
            .into_iter()
            .map(|e| (e.source().index() as i32, e.target().index() as i32))
            .unzip();

        let indices = self.stack.pop().unwrap()
            .into_iter()
            .map(|n| n.index() as i32)
            .collect();

        (nodes, sources, targets, indices)
    }
}

impl Visitor for GraphBuilder {
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
            Constant(functor) => self.visit_functor(functor),
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
    #[pyfn(module, "parse")]
    fn parse<'p>(py: Python<'p>, byte_object: &PyBytes) -> PyResult<Vec<&'p PyBytes>> {
        let bytes = byte_object.as_bytes();
        let mut parsed = vec![];
        let mut parser = tptp_input_iterator::<()>(bytes);
        for input in &mut parser {
            parsed.push(PyBytes::new(py, format!("{}", input).as_ref()));
        }
        if !parser.finish().is_ok() {
            Err(pyo3::exceptions::SyntaxError::py_err(""))
        } else {
            Ok(parsed)
        }
    }

    #[pyfn(module, "graph")]
    fn graph<'p>(py: Python<'p>, selected_list: &PyList, action_list: &PyList) -> PyResult<(&'p PyArray1<i32>, &'p PyArray1<i32>, &'p PyArray1<i32>, &'p PyArray1<i32>)> {
        let mut selected_clauses: Vec<&PyBytes> = vec![];
        for object in selected_list {
            selected_clauses.push(object.extract()?);
        }
        let mut action_clauses: Vec<&PyBytes> = vec![];
        for object in action_list {
            action_clauses.push(object.extract()?);
        }

        let mut builder = GraphBuilder::default();
        builder.level();
        for clause in selected_clauses {
            if let Ok((_, clause)) = cnf_annotated::<()>(clause.as_bytes()) {
                builder.visit(clause, true);
            } else {
                return Err(pyo3::exceptions::SyntaxError::py_err(""));
            }
        }
        for clause in action_clauses {
            if let Ok((_, clause)) = cnf_annotated::<()>(clause.as_bytes()) {
                builder.visit(clause, false);
            } else {
                return Err(pyo3::exceptions::SyntaxError::py_err(""));
            }
        }

        let (nodes, sources, targets, clauses) = builder.finish();
        let nodes = PyArray::from_vec(py, nodes);
        let sources = PyArray::from_vec(py, sources);
        let targets = PyArray::from_vec(py, targets);
        let clauses = PyArray::from_vec(py, clauses);
        Ok((nodes, sources, targets, clauses))
    }

    Ok(())
}
