//! Small, typed predicate parsers shared by planner-facing index rules.

use std::collections::HashMap;

use datafusion::common::ScalarValue;
use datafusion::logical_expr::{Expr, Operator};

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct InclusiveIntRange {
    pub lower: Option<i64>,
    pub upper: Option<i64>,
}

impl InclusiveIntRange {
    pub fn overlaps(&self, minimum: i64, maximum: i64) -> bool {
        self.lower.is_none_or(|lower| maximum >= lower)
            && self.upper.is_none_or(|upper| minimum <= upper)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HalfOpenIntRange {
    pub column: String,
    pub lower: i64,
    pub upper: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IntComparison {
    pub column: String,
    pub operator: Operator,
    pub value: i64,
}

pub fn half_open_int_range(predicate: &Expr) -> Option<HalfOpenIntRange> {
    let mut terms = Vec::new();
    conjuncts(predicate, &mut terms);
    let mut column = None;
    let mut lower = None;
    let mut upper = None;
    for term in terms {
        let comparison = int_comparison(term)?;
        if column
            .as_ref()
            .is_some_and(|column| column != &comparison.column)
        {
            return None;
        }
        column = Some(comparison.column);
        match comparison.operator {
            Operator::GtEq if lower.replace(comparison.value).is_none() => {}
            Operator::Lt if upper.replace(comparison.value).is_none() => {}
            _ => return None,
        }
    }
    let range = HalfOpenIntRange {
        column: column?,
        lower: lower?,
        upper: upper?,
    };
    (range.lower < range.upper).then_some(range)
}

/// Inclusive integer ranges implied by top-level conjuncts.
///
/// Strict comparisons are widened to include their boundary. Callers use these
/// ranges only to skip disjoint segment summaries, so widening preserves scan
/// correctness while still excluding segments outside bounded time windows.
pub fn int_column_ranges(filters: &[Expr]) -> HashMap<String, InclusiveIntRange> {
    let mut ranges = HashMap::new();
    for filter in filters {
        collect_int_ranges(filter, &mut ranges);
    }
    ranges
}

fn collect_int_ranges(expr: &Expr, ranges: &mut HashMap<String, InclusiveIntRange>) {
    if let Expr::BinaryExpr(binary) = expr {
        if binary.op == Operator::And {
            collect_int_ranges(&binary.left, ranges);
            collect_int_ranges(&binary.right, ranges);
            return;
        }
    }
    let Some(comparison) = int_comparison(expr) else {
        return;
    };
    let range = ranges.entry(comparison.column).or_default();
    match comparison.operator {
        Operator::Eq => {
            tighten_lower(&mut range.lower, comparison.value);
            tighten_upper(&mut range.upper, comparison.value);
        }
        Operator::Gt | Operator::GtEq => tighten_lower(&mut range.lower, comparison.value),
        Operator::Lt | Operator::LtEq => tighten_upper(&mut range.upper, comparison.value),
        _ => {}
    }
}

fn tighten_lower(bound: &mut Option<i64>, candidate: i64) {
    *bound = Some(bound.map_or(candidate, |current| current.max(candidate)));
}

fn tighten_upper(bound: &mut Option<i64>, candidate: i64) {
    *bound = Some(bound.map_or(candidate, |current| current.min(candidate)));
}

pub fn conjuncts<'a>(expr: &'a Expr, output: &mut Vec<&'a Expr>) {
    match expr {
        Expr::BinaryExpr(binary) if binary.op == Operator::And => {
            conjuncts(&binary.left, output);
            conjuncts(&binary.right, output);
        }
        expr => output.push(expr),
    }
}

pub(crate) fn int_comparison(expr: &Expr) -> Option<IntComparison> {
    let Expr::BinaryExpr(binary) = expr else {
        return None;
    };
    match (binary.left.as_ref(), binary.right.as_ref()) {
        (Expr::Column(column), literal) => Some(IntComparison {
            column: column.name.clone(),
            operator: binary.op,
            value: int_literal(literal)?,
        }),
        (literal, Expr::Column(column)) => {
            let operator = match binary.op {
                Operator::Lt => Operator::Gt,
                Operator::LtEq => Operator::GtEq,
                Operator::Gt => Operator::Lt,
                Operator::GtEq => Operator::LtEq,
                _ => return None,
            };
            Some(IntComparison {
                column: column.name.clone(),
                operator,
                value: int_literal(literal)?,
            })
        }
        _ => None,
    }
}

fn int_literal(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Literal(ScalarValue::Int64(Some(value)), _) => Some(*value),
        Expr::Cast(cast) => int_literal(&cast.expr),
        _ => None,
    }
}
