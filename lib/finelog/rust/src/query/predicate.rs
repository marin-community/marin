//! Small, typed predicate parsers shared by planner-facing index rules.

use std::collections::HashMap;

use datafusion::arrow::datatypes::DataType;
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
        if let Some((column, lower, upper)) = float_cast_range(expr) {
            let range = ranges.entry(column).or_default();
            if let Some(lower) = lower {
                tighten_lower(&mut range.lower, lower);
            }
            if let Some(upper) = upper {
                tighten_upper(&mut range.upper, upper);
            }
        }
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

/// Below this magnitude every integer is exact in an `f64`.
const F64_EXACT_INTEGER_LIMIT: f64 = 9_007_199_254_740_992.0;

/// An integer column compared as `Float64` with a float literal, the form
/// `epoch_ms > extract(epoch from now()) * 1000 - 600000` folds to, as inclusive
/// integer bounds on the column: the floor of the literal for a lower bound and
/// its ceiling for an upper bound. That never excludes a matching row, so it is
/// for segment pruning only, like the rest of [`int_column_ranges`]; exact
/// planners keep using [`int_comparison`].
fn float_cast_range(expr: &Expr) -> Option<(String, Option<i64>, Option<i64>)> {
    let Expr::BinaryExpr(binary) = expr else {
        return None;
    };
    let (column, operator, value) = match (
        float_cast_column(&binary.left),
        float_literal(&binary.right),
    ) {
        (Some(column), Some(value)) => (column, binary.op, value),
        _ => {
            let (Some(value), Some(column)) = (
                float_literal(&binary.left),
                float_cast_column(&binary.right),
            ) else {
                return None;
            };
            let operator = match binary.op {
                Operator::Lt => Operator::Gt,
                Operator::LtEq => Operator::GtEq,
                Operator::Gt => Operator::Lt,
                Operator::GtEq => Operator::LtEq,
                Operator::Eq => Operator::Eq,
                _ => return None,
            };
            (column, operator, value)
        }
    };
    // Past 2^53 the cast column itself rounds, so floor and ceiling no longer
    // bound the integers that compare true.
    if !value.is_finite() || value.abs() >= F64_EXACT_INTEGER_LIMIT {
        return None;
    }
    let lower = value.floor() as i64;
    let upper = value.ceil() as i64;
    match operator {
        Operator::Eq => Some((column, Some(lower), Some(upper))),
        Operator::Gt | Operator::GtEq => Some((column, Some(lower), None)),
        Operator::Lt | Operator::LtEq => Some((column, None, Some(upper))),
        _ => None,
    }
}

fn float_cast_column(expr: &Expr) -> Option<String> {
    let (inner, data_type) = match expr {
        Expr::Cast(cast) => (cast.expr.as_ref(), &cast.data_type),
        Expr::TryCast(cast) => (cast.expr.as_ref(), &cast.data_type),
        _ => return None,
    };
    let Expr::Column(column) = inner else {
        return None;
    };
    (*data_type == DataType::Float64).then(|| column.name.clone())
}

fn float_literal(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Literal(ScalarValue::Float64(Some(value)), _) => Some(*value),
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

#[cfg(test)]
mod tests {
    use datafusion::prelude::{cast, col, lit};

    use super::*;

    fn epoch_as_float() -> Expr {
        cast(col("epoch_ms"), DataType::Float64)
    }

    fn range(filter: Expr) -> Option<InclusiveIntRange> {
        int_column_ranges(&[filter]).remove("epoch_ms")
    }

    #[test]
    fn a_float_cast_comparison_widens_to_the_enclosing_integers() {
        assert_eq!(
            range(epoch_as_float().gt(lit(10.5_f64))),
            Some(InclusiveIntRange {
                lower: Some(10),
                upper: None
            })
        );
        assert_eq!(
            range(epoch_as_float().lt_eq(lit(10.5_f64))),
            Some(InclusiveIntRange {
                lower: None,
                upper: Some(11)
            })
        );
        assert_eq!(
            range(epoch_as_float().eq(lit(10.0_f64))),
            Some(InclusiveIntRange {
                lower: Some(10),
                upper: Some(10)
            })
        );
    }

    #[test]
    fn a_literal_on_the_left_flips_the_operator() {
        assert_eq!(
            range(lit(20.25_f64).gt(epoch_as_float())),
            Some(InclusiveIntRange {
                lower: None,
                upper: Some(21)
            })
        );
    }

    #[test]
    fn float_and_integer_terms_on_one_column_combine() {
        let filter = epoch_as_float()
            .gt_eq(lit(100.5_f64))
            .and(col("epoch_ms").lt(lit(200_i64)));
        assert_eq!(
            range(filter),
            Some(InclusiveIntRange {
                lower: Some(100),
                upper: Some(200)
            })
        );
    }

    #[test]
    fn non_finite_huge_or_non_float64_comparisons_give_no_range() {
        assert_eq!(range(epoch_as_float().gt(lit(f64::NAN))), None);
        assert_eq!(range(epoch_as_float().gt(lit(f64::INFINITY))), None);
        assert_eq!(range(epoch_as_float().gt(lit(1e17_f64))), None);
        assert_eq!(
            range(cast(col("epoch_ms"), DataType::Float32).gt(lit(10.5_f64))),
            None
        );
        assert_eq!(range(epoch_as_float().not_eq(lit(10.5_f64))), None);
    }

    #[test]
    fn exact_parsers_still_reject_the_float_form() {
        assert_eq!(int_comparison(&epoch_as_float().gt(lit(10.5_f64))), None);
        let half_open = epoch_as_float()
            .gt_eq(lit(10.0_f64))
            .and(epoch_as_float().lt(lit(20.0_f64)));
        assert_eq!(half_open_int_range(&half_open), None);
    }
}
