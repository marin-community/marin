# Question
Title: Generate field in MySQL SELECT
If I've got a table containing Field1 and Field2 can I generate a new field in the select statement? For example, a normal query would be: 
```
SELECT Field1, Field2 FROM Table

```
And I want to also create Field3 and have that returned in the resultset... something along the lines of this would be ideal: 
```
SELECT Field1, Field2, Field3 = 'Value' FROM Table

```
Is this possible at all?

# Answer
> 12 votes
```
SELECT Field1, Field2, 'Value' Field3 FROM Table

```

or for clarity

```
SELECT Field1, Field2, 'Value' AS Field3 FROM Table

```

# Answer
> 5 votes
Yes - it's very possible, in fact you almost had it! Try:

```
SELECT Field1, Field2, 'Value' AS `Field3` FROM Table

```

---
Tags: sql, mysql
---