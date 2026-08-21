/** Quote a string literal for Finelog's SQL API, which has no parameter binding. */
export function sqlString(value: string): string {
  return `'${value.replace(/'/g, "''")}'`
}
