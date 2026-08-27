//! Small, typed predicate parsers shared by planner-facing index rules.

use datafusion::common::ScalarValue;
use datafusion::logical_expr::{Expr, Operator};

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
