"""AST node definitions for ChimeraLang."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chimera.tokens import SourceSpan


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class ASTNode:
    """ASTNode."""
    span: SourceSpan | None = field(default=None, repr=False, kw_only=True)


# ---------------------------------------------------------------------------
# Type Expressions
# ---------------------------------------------------------------------------

@dataclass
class TypeExpr(ASTNode):
    """Base for all type expressions."""


@dataclass
class PrimitiveType(TypeExpr):
    """PrimitiveType."""
    name: str  # Int, Float, Bool, Text, Void


@dataclass
class ProbabilisticType(TypeExpr):
    """Confident<T>, Explore<T>, Converge<T>, Provisional<T>"""
    wrapper: str  # Confident | Explore | Converge | Provisional
    inner: TypeExpr


@dataclass
class MemoryType(TypeExpr):
    """Ephemeral<T>, Persistent<T>"""
    scope: str  # Ephemeral | Persistent
    inner: TypeExpr


@dataclass
class GenericType(TypeExpr):
    """List<T>, Map<K,V>, Option<T>, Result<T,E>"""
    name: str
    params: list[TypeExpr]


@dataclass
class NamedType(TypeExpr):
    """User-defined type reference."""
    name: str


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

@dataclass
class Expr(ASTNode):
    """Base for all expressions."""


@dataclass
class IntLiteral(Expr):
    """IntLiteral."""
    value: int


@dataclass
class FloatLiteral(Expr):
    """FloatLiteral."""
    value: float


@dataclass
class StringLiteral(Expr):
    """StringLiteral."""
    value: str


@dataclass
class BoolLiteral(Expr):
    """BoolLiteral."""
    value: bool


@dataclass
class ListLiteral(Expr):
    """ListLiteral."""
    elements: list[Expr]


@dataclass
class Identifier(Expr):
    """Identifier."""
    name: str


@dataclass
class BinaryOp(Expr):
    """BinaryOp."""
    op: str
    left: Expr
    right: Expr


@dataclass
class UnaryOp(Expr):
    """UnaryOp."""
    op: str
    operand: Expr


@dataclass
class CallExpr(Expr):
    """CallExpr."""
    callee: Expr
    args: list[Expr]


@dataclass
class MemberExpr(Expr):
    """MemberExpr."""
    obj: Expr
    member: str


@dataclass
class IfExpr(Expr):
    """IfExpr."""
    condition: Expr
    then_body: list[Statement]
    else_body: list[Statement] | None = None


@dataclass
class CompareChain(Expr):
    """quality: security > performance > readability"""
    items: list[str]


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

@dataclass
class Statement(ASTNode):
    """Base for all statements."""


@dataclass
class ValDecl(Statement):
    """ValDecl."""
    name: str
    type_ann: TypeExpr | None = None
    value: Expr | None = None


@dataclass
class ReturnStmt(Statement):
    """ReturnStmt."""
    value: Expr | None = None


@dataclass
class ExprStmt(Statement):
    """ExprStmt."""
    expr: Expr


@dataclass
class AssertStmt(Statement):
    """AssertStmt."""
    condition: Expr
    message: str | None = None


@dataclass
class EmitStmt(Statement):
    """EmitStmt."""
    value: Expr


@dataclass
class ForStmt(Statement):
    """for x in collection ... end"""
    target: str
    iterable: Expr
    body: list[Statement]


@dataclass
class MatchArm(ASTNode):
    """One arm of a match expression: | pattern => body"""
    pattern: Expr | None  # None = wildcard (_)
    body: list[Statement]


@dataclass
class MatchExpr(Expr):
    """match subject | p1 => ... | p2 => ... end"""
    subject: Expr
    arms: list[MatchArm]


# ---------------------------------------------------------------------------
# Constraint Blocks (inside fn / gate / goal / reason)
# ---------------------------------------------------------------------------

@dataclass
class Constraint(ASTNode):
    """Base for must/allow/forbidden."""


@dataclass
class MustConstraint(Constraint):
    """MustConstraint."""
    expr: Expr


@dataclass
class AllowConstraint(Constraint):
    """AllowConstraint."""
    capabilities: list[str]


@dataclass
class ForbiddenConstraint(Constraint):
    """ForbiddenConstraint."""
    capabilities: list[str]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class Param(ASTNode):
    """Param."""
    name: str
    type_ann: TypeExpr


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

@dataclass
class Declaration(ASTNode):
    """Base for top-level declarations."""


@dataclass
class FnDecl(Declaration):
    """FnDecl."""
    name: str
    params: list[Param]
    return_type: TypeExpr | None = None
    constraints: list[Constraint] = field(default_factory=list)
    body: list[Statement] = field(default_factory=list)


@dataclass
class GateDecl(Declaration):
    """GateDecl."""
    name: str
    params: list[Param]
    return_type: TypeExpr | None = None
    branches: int = 3
    collapse: str = "majority"
    threshold: float = 0.85
    fallback: str = "escalate"
    body: list[Statement] = field(default_factory=list)


@dataclass
class GoalDecl(Declaration):
    """GoalDecl."""
    description: str
    constraints_list: list[str] = field(default_factory=list)
    quality_axes: list[str] = field(default_factory=list)
    explore_budget: float = 1.0
    body: list[Statement] = field(default_factory=list)


@dataclass
class ReasonDecl(Declaration):
    """ReasonDecl."""
    name: str
    params: list[Param]
    return_type: TypeExpr | None = None
    given: list[str] = field(default_factory=list)
    explore_expr: Expr | None = None
    evaluate_expr: Expr | None = None
    commit_strategy: str = "highest_consensus"
    body: list[Statement] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Program (root node)
# ---------------------------------------------------------------------------

@dataclass
class Program(ASTNode):
    """Program."""
    declarations: list[Declaration | Statement] = field(default_factory=list)
