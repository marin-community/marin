/*
 *  cool.y
 *              Parser definition for the COOL language.
 *
 */
%{
#include "cool-tree.h"
#include "stringtab.h"
#include "utilities.h"

/* Set the size of the parser stack to be sufficient large to accomodate
   our tests.  There seems to be some problem with Bison's dynamic
   reallocation of the parser stack, as it should resize to a max
   size of YYMAXDEPTH, which is by default 10000. */
#define YYINITDEPTH 3000

/* Locations */
#define YYLTYPE int              /* the type of locations */
#define cool_yylloc curr_lineno  /* use the curr_lineno from the lexer
                                    for the location of tokens */
extern int node_lineno;          /* set before constructing a tree node
                                    to whatever you want the line number
                                    for the tree node to be */

/* The default action for locations.  Use the location of the first
   terminal/non-terminal and set the node_lineno to that value. */
#define YYLLOC_DEFAULT(Current, Rhs, N)		  \
  Current = (Rhs)[1];                             \
  node_lineno = Current;

#define SET_NODELOC(Current)			\
  node_lineno = Current;

/* IMPORTANT NOTE ON LINE NUMBERS
*********************************
* The above definitions and macros cause every terminal in your grammar to
* have the line number supplied by the lexer. The only task you have to
* implement for line numbers to work correctly, is to use SET_NODELOC()
* before constructing any constructs from non-terminals in your grammar.
* Example: Consider you are matching on the following very restrictive
* (fictional) construct that matches a plus between two integer constants.
* (SUCH A RULE SHOULD NOT BE  PART OF YOUR PARSER):

plus_consts : INT_CONST '+' INT_CONST

* where INT_CONST is a terminal for an integer constant. Now, a correct
* action for this rule that attaches the correct line number to plus_const
* would look like the following:

plus_consts : INT_CONST '+' INT_CONST
{
// Set the line number of the current non-terminal:
// ***********************************************
// You can access the line numbers of the i'th item with @i, just
// like you acess the value of the i'th exporession with $i.
//
// Here, we choose the line number of the last INT_CONST (@3) as the
// line number of the resulting expression (@$). You are free to pick
// any reasonable line as the line number of non-terminals. If you
* omit the statement @$=..., bison has default rules for deciding which
* line number to use. Check the manual for details if you are interested.
@$ = @3;


// Observe that we call SET_NODELOC(@3); this will set the global variable
// node_lineno to @3. Since the constructor call "plus" uses the value of
// this global, the plus node will now have the correct line number.
SET_NODELOC(@3);

// construct the result node:
$$ = plus(int_const($1), int_const($3));
}

*/

extern char *curr_filename;

void yyerror(const char *s);  /*  defined below; called for each parse error */
extern int yylex();           /*  the entry point to the lexer  */
Program ast_root;	      /* the result of the parse  */
Classes parse_results;        /* for use in semantic analysis */
int omerrs = 0;               /* number of erros in lexing and parsing */
%}

/* A union of all the types that can be the result of parsing actions. */
%union {
  bool boolean;
  Symbol symbol;
  Program program;
  Class_ class_;
  Classes classes;
  Feature feature;
  Features features;
  Formal formal;
  Formals formals;
  Case case_;
  Cases cases;
  Expression expression;
  Expressions expressions;
  LetBinding let_binding_val;
  LetBindings let_bindings_val;
  const char *error_msg;
}

/*
   Declare the terminals; a few have types for associated lexemes.
   The token ERROR is never used in the parser; thus, it is a parse
   error when the lexer returns it.

   The integer following token declaration is the numeric constant used
   to represent that token internally.  Typically, Bison generates these
   on its own, but we give explicit numbers to prevent version parity
   problems (bison 1.25 and earlier start at 258, later versions -- at
   257)

   NOTE: these numbers need to match what's in cool-parse.h !
 */
%token CLASS 258 ELSE 259 FI 260 IF 261 IN 262
%token INHERITS 263 LET 264 LOOP 265 POOL 266 THEN 267 WHILE 268
%token CASE 269 ESAC 270 OF 271 DARROW 272 NEW 273 ISVOID 274
%token <symbol>  STR_CONST 275 INT_CONST 276
%token <boolean> BOOL_CONST 277
%token <symbol>  TYPEID 278 OBJECTID 279
%token ASSIGN 280 NOT 281 LE 282 ERROR 283

/*  DON'T CHANGE ANYTHING ABOVE THIS LINE, OR YOUR PARSER WONT WORK       */
/**************************************************************************/

/* Complete the nonterminal list below, giving a type for the semantic
   value of each non terminal. (See section 3.6 in the bison 
   documentation for details). */

/* Declare types for the grammar's non-terminals. */
%type <program> program
%type <classes> class_list
%type <class_> class
%type <features> optional_feature_list
%type <features> feature_list
%type <feature> feature
%type <formals> formal_list
%type <formal> formal
%type <expressions> expr_list
%type <expression> expr
%type <cases> case_list
%type <case_> case
%type <expressions> block_expr
%type <expression> let_expr
%type <let_binding_val> let_binding
%type <let_bindings_val> let_binding_list

/* Precedence declarations go here. */

%right ASSIGN
%left NOT
%nonassoc LE '<' '='
%left '+' '-'
%left '*' '/'
%left ISVOID
%left '~'
%left '@'
%left '.'


%%
// Save the root of the abstract syntax tree in a global variable.
program	: class_list	{ @$ = @1; ast_root = program($1); }

class_list:
  class	
  { $$ = single_Classes($1);
    parse_results = $$; }
| class_list class
  { $$ = append_Classes($1,single_Classes($2));
    parse_results = $$; }
;

/* If no parent is specified, the class inherits from the Object class. */
class	: CLASS TYPEID '{' optional_feature_list '}' ';'
{ $$ = class_($2,idtable.add_string("Object"),$4,
	      stringtable.add_string(curr_filename)); }
| CLASS TYPEID INHERITS TYPEID '{' optional_feature_list '}' ';'
{ $$ = class_($2,$4,$6,stringtable.add_string(curr_filename)); }
;

/* Feature list may be empty, but no empty features in list. */
optional_feature_list:		
  /* empty */
  {  $$ = nil_Features(); }
| feature_list
  {  $$ = $1; }
;

feature_list
: feature ';'
  {  $$ = single_Features($1); }
| feature_list feature ';'
  {  $$ = append_Features($1,single_Features($2)); }
;

feature:
  OBJECTID '(' formal_list ')' ':' TYPEID '{' expr '}'
  { $$ = method($1,$3,$6, $8); }
  | OBJECTID ':' TYPEID
  { $$ = attr($1,$3,no_expr()); }
  | OBJECTID ':' TYPEID ASSIGN expr
  { $$ = attr($1,$3,$5); }
  | error ';' {}
  ;

formal_list:
  /* empty */
  {  $$ = nil_Formals(); }
| formal
  {  $$ = single_Formals($1); }
| formal_list ',' formal
  {  $$ = append_Formals($1,single_Formals($3)); }
;

formal:
  OBJECTID ':' TYPEID
  { $$ = formal($1,$3); }
  | OBJECTID ':' TYPEID '[' ']'
  { $$ = formal($1,$3); }
  ;

expr_list:
  expr ';'
  { $$ = single_Expressions($1); }
  | expr_list expr ';'
  { $$ = append_Expressions($1, single_Expressions($2)); }
  | expr_list error ';' /* Skip bad expression */
;

block_expr:
  expr ';'
  { $$ = single_Expressions($1); }
  | block_expr expr ';'
  { $$ = append_Expressions($1, single_Expressions($2)); }
  | block_expr error ';'
  { $$ = $1; }
;

case_list:
  case
  { $$ = single_Cases($1); }
  | case_list case
  { $$ = append_Cases($1, single_Cases($2)); }
;

case:
  OBJECTID ':' TYPEID DARROW expr ';'
  { $$ = branch($1, $3, $5); }
;

expr:
  OBJECTID ASSIGN expr
  { $$ = assign($1, $3); }
  | expr '.' OBJECTID '(' expr_list ')'
  { $$ = dispatch($1,$3,$5); }
  | expr '@' TYPEID '.' OBJECTID '(' expr_list ')'
  { $$ = static_dispatch($1,$3,$5,$7); }
  | OBJECTID '(' expr_list ')'
  { $$ = dispatch(object(idtable.add_string("self")), $1, $3); }
  | IF expr THEN expr ELSE expr FI
  { $$ = cond($2, $4, $6); }
  | WHILE expr LOOP expr POOL
  { $$ = loop($2, $4); }
  | '{' block_expr '}'
  { $$ = block($2); }
  | LET let_expr
  { $$ = $2; }
  | CASE expr OF case_list ESAC
  { $$ = typcase($2, $4); }
  | NEW TYPEID
  { $$ = new_($2); }
  | expr '+' expr
  { $$ = plus($1, $3); }
  
  | expr '-' expr
  { $$ = sub($1, $3); }
  
  | expr '*' expr
  { $$ = mul($1, $3); }
  
  | expr '/' expr
  { $$ = divide($1, $3); }
  
  | '~' expr
  { $$ = neg($2); }

  | expr '<' expr
  { $$ = lt($1, $3); }
  
  | expr LE expr
  { $$ = leq($1, $3); }
  
  | expr '=' expr
  { $$ = eq($1, $3); }
  
  | NOT expr
  { $$ = comp($2); }

  | ISVOID expr
  { $$ = isvoid($2); }

  | INT_CONST
  { $$ = int_const($1); }
  
  | STR_CONST
  { $$ = string_const($1); }
  
  | BOOL_CONST
  { $$ = bool_const($1); }
  
  | OBJECTID
  { $$ = object($1); }

  | '(' expr ')'
  { $$ = $2; }

let_expr:
  OBJECTID ':' TYPEID IN expr
  { $$ = let($1, $3, no_expr(), $5); }
  | OBJECTID ':' TYPEID ASSIGN expr IN expr
  { $$ = let($1, $3, $5, $7); }
  | OBJECTID ':' TYPEID ',' let_expr
  { $$ = let($1, $3, no_expr(), $5); }
  | OBJECTID ':' TYPEID ASSIGN expr ',' let_expr
  { $$ = let($1, $3, $5, $7); }
  | error ',' let_expr
  { $$ = $3; }
  | error IN expr
  { $$ = $3; }
  | let_binding_list ',' error /* Skip bad binding after comma */
  | error /* Skip bad first binding */
;

let_binding_list:
    let_binding
    { $$ = single_LetBindings($1); }
  | let_binding_list ',' let_binding
    { $$ = append_LetBindings($1, single_LetBindings($3)); }
  | let_binding_list ',' error { $$ = $1; /* Keep previous valid bindings */ }
  | error { $$ = nil_LetBindings(); /* Start with empty list if first is bad */ }
;

let_binding:
  OBJECTID ':' TYPEID
  { $$ = let_binding($1, $3, no_expr()); }
  | OBJECTID ':' TYPEID ASSIGN expr
  { $$ = let_binding($1, $3, $5); }
;

/* end of grammar */
%%

/* This function is called automatically when Bison detects a parse error. */
void yyerror(const char *s)
{
  extern int curr_lineno;

  std::cerr << "\"" << curr_filename << "\", line " << curr_lineno << ": " \
    << s << " at or near ";
  print_cool_token(yychar);
  std::cerr << std::endl;
  omerrs++;
  if(omerrs>50) { std::cerr << "More than 50 errors" << std::endl; exit(1);}
}
