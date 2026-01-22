#!/usr/bin/env python3
"""
APL to Python Transpiler

Transpiles APL source code to Python using NumPy for array operations.
Works with decompiled APL from ANS files or direct APL source.

Usage:
    python apl_transpiler.py SQL.ans                    # From ANS file
    python apl_transpiler.py workspace.apl              # From APL source
    python apl_transpiler.py SQL.ans --python-only      # Skip APL output
    python apl_transpiler.py SQL.ans -o mydir           # Custom output dir

Architecture:
    ANS File → ANSDecompiler → APL Source → APLLexer → Tokens → APLParser → AST → PythonGenerator → .py
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# =============================================================================
# Token Types
# =============================================================================

class TokenType(Enum):
    """APL token types"""
    NUMBER = auto()      # 123, ¯45, 3.14, 1E10
    STRING = auto()      # 'hello'
    NAME = auto()        # VAR, FUNC, _temp
    PRIMITIVE = auto()   # ⍴, ⍳, +, -, ×, ÷, etc.
    OPERATOR = auto()    # /, \, ¨, ⍨, ⍣, ∘, ⍤
    ASSIGN = auto()      # ←
    BRANCH = auto()      # →
    LPAREN = auto()      # (
    RPAREN = auto()      # )
    LBRACKET = auto()    # [
    RBRACKET = auto()    # ]
    LBRACE = auto()      # {
    RBRACE = auto()      # }
    SEMICOLON = auto()   # ;
    COLON = auto()       # :
    DIAMOND = auto()     # ⋄ (statement separator)
    NEWLINE = auto()
    COMMENT = auto()     # ⍝ ...
    LABEL = auto()       # L1:
    QUAD = auto()        # ⎕
    QUOTE_QUAD = auto()  # ⍞
    ZILDE = auto()       # ⍬
    DEL = auto()         # ∇
    ALPHA = auto()       # ⍺
    OMEGA = auto()       # ⍵
    SYSTEM_VAR = auto()  # ⎕IO, ⎕CT, etc.
    AXIS = auto()        # [1], [2] - axis specification
    EOF = auto()


@dataclass
class Token:
    """Represents a single token from APL source"""
    type: TokenType
    value: Any
    line: int
    column: int

    def __repr__(self):
        return f"Token({self.type.name}, {repr(self.value)}, {self.line}:{self.column})"


# =============================================================================
# APL Character Sets
# =============================================================================

# APL primitive functions (can be monadic or dyadic)
APL_PRIMITIVES = set(
    '+-×÷|⌊⌈*⍟○!?~∧∨⍲⍱<≤=≥>≠⍴⍳↑↓⊂⊃⊆⊇∩∪⍷∊⌽⊖⍉⌹⍋⍒⍎⍕,⍪'
    '⊣⊢⌷≡≢⍸'
)

# APL operators (modify functions)
APL_OPERATORS = set('/\\⌿⍀¨⍨⍣∘⍤⍥.')

# Characters that can start a name
NAME_START = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_∆⍙')
NAME_CHARS = NAME_START | set('0123456789')


# =============================================================================
# AST Node Definitions
# =============================================================================

@dataclass
class ASTNode:
    """Base class for all AST nodes"""
    pass


@dataclass
class NumberLiteral(ASTNode):
    """Numeric literal"""
    value: Union[int, float, List[Union[int, float]]]


@dataclass
class StringLiteral(ASTNode):
    """String literal"""
    value: str


@dataclass
class Variable(ASTNode):
    """Variable reference"""
    name: str


@dataclass
class SystemVariable(ASTNode):
    """System variable like ⎕IO"""
    name: str


@dataclass
class Zilde(ASTNode):
    """Empty numeric vector ⍬"""
    pass


@dataclass
class Assignment(ASTNode):
    """Variable assignment: X←Y"""
    target: str
    value: ASTNode
    is_indexed: bool = False
    indices: Optional[ASTNode] = None


@dataclass
class MonadicCall(ASTNode):
    """Monadic function call: fX"""
    func: str
    arg: ASTNode


@dataclass
class DyadicCall(ASTNode):
    """Dyadic function call: XfY"""
    func: str
    left: ASTNode
    right: ASTNode


@dataclass
class OperatorExpr(ASTNode):
    """Operator expression: f/X, f\\X, f¨X"""
    operator: str
    func: Union[str, ASTNode]
    operand: ASTNode
    is_dyadic: bool = False
    left_operand: Optional[ASTNode] = None


@dataclass
class IndexExpr(ASTNode):
    """Array indexing: X[I] or X[I;J]"""
    array: ASTNode
    indices: List[Optional[ASTNode]]  # None means "all" for that axis


@dataclass
class Branch(ASTNode):
    """Branch expression: →L or →condition/label"""
    target: ASTNode


@dataclass
class Label(ASTNode):
    """Label definition: L1:"""
    name: str


@dataclass
class QuadOutput(ASTNode):
    """Quad output: ⎕←X"""
    value: ASTNode


@dataclass
class QuadInput(ASTNode):
    """Quad input: ⎕"""
    pass


@dataclass
class QuoteQuadInput(ASTNode):
    """Quote quad input: ⍞"""
    pass


@dataclass
class Strand(ASTNode):
    """Vector strand: 1 2 3 or A B C"""
    elements: List[ASTNode]


@dataclass
class ParenExpr(ASTNode):
    """Parenthesized expression"""
    expr: ASTNode


@dataclass
class DfnLambda(ASTNode):
    """Direct function (dfn): {⍺+⍵}"""
    body: str
    lines: List[str]


@dataclass
class Comment(ASTNode):
    """Comment: ⍝ text"""
    text: str


@dataclass
class FunctionDef(ASTNode):
    """Complete function definition"""
    name: str
    header: str
    result_var: Optional[str]
    left_arg: Optional[str]
    right_arg: Optional[str]
    locals: List[str]
    body: List[ASTNode]
    raw_lines: List[str]


@dataclass
class Statement(ASTNode):
    """A single statement (may contain label)"""
    label: Optional[str]
    expr: Optional[ASTNode]
    comment: Optional[str] = None


@dataclass
class Program(ASTNode):
    """Top-level program with multiple functions"""
    functions: List[FunctionDef]
    variables: Dict[str, ASTNode]


# =============================================================================
# APL Lexer
# =============================================================================

class APLLexer:
    """
    Tokenizes APL source code into a stream of tokens.

    Handles:
    - APL special characters (⍳, ⍴, ←, etc.)
    - High-minus (¯) for negative numbers
    - String literals with doubled quotes
    - Comments (⍝ to end of line)
    - Function headers and locals
    """

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []

    def current_char(self) -> Optional[str]:
        """Get current character or None if at end"""
        if self.pos < len(self.source):
            return self.source[self.pos]
        return None

    def peek_char(self, offset: int = 1) -> Optional[str]:
        """Look ahead without advancing"""
        pos = self.pos + offset
        if pos < len(self.source):
            return self.source[pos]
        return None

    def advance(self) -> Optional[str]:
        """Move to next character"""
        char = self.current_char()
        if char:
            self.pos += 1
            if char == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
        return char

    def add_token(self, token_type: TokenType, value: Any = None):
        """Add a token to the list"""
        self.tokens.append(Token(
            type=token_type,
            value=value,
            line=self.line,
            column=self.column
        ))

    def skip_whitespace(self):
        """Skip spaces and tabs (not newlines)"""
        while self.current_char() in ' \t':
            self.advance()

    def read_number(self) -> Token:
        """Read a numeric literal"""
        start_col = self.column
        start_line = self.line
        result = ""

        # Handle high-minus for negative
        if self.current_char() == '¯':
            result = '-'
            self.advance()

        # Integer part
        while self.current_char() and self.current_char() in '0123456789':
            result += self.advance()

        # Decimal part
        if self.current_char() == '.' and self.peek_char() in '0123456789':
            result += self.advance()  # the dot
            while self.current_char() and self.current_char() in '0123456789':
                result += self.advance()

        # Exponent
        if self.current_char() and self.current_char() in 'Ee':
            result += self.advance()
            if self.current_char() == '¯':
                result += '-'
                self.advance()
            elif self.current_char() in '+-':
                result += self.advance()
            while self.current_char() and self.current_char() in '0123456789':
                result += self.advance()

        # Convert to appropriate type
        try:
            if '.' in result or 'e' in result.lower():
                value = float(result)
            else:
                value = int(result)
        except ValueError:
            value = result

        return Token(TokenType.NUMBER, value, start_line, start_col)

    def read_string(self) -> Token:
        """Read a string literal"""
        start_col = self.column
        start_line = self.line
        result = ""

        self.advance()  # Opening quote

        while True:
            char = self.current_char()
            if char is None:
                break
            if char == "'":
                self.advance()
                # Doubled quote is escaped quote
                if self.current_char() == "'":
                    result += "'"
                    self.advance()
                else:
                    break
            else:
                result += char
                self.advance()

        return Token(TokenType.STRING, result, start_line, start_col)

    def read_name(self) -> Token:
        """Read an identifier/name"""
        start_col = self.column
        start_line = self.line
        result = ""

        while self.current_char() and self.current_char() in NAME_CHARS:
            result += self.advance()

        # Check for label (name followed by colon at start of line)
        # This is handled differently - we just return the name
        return Token(TokenType.NAME, result, start_line, start_col)

    def read_comment(self) -> Token:
        """Read a comment (⍝ to end of line)"""
        start_col = self.column
        start_line = self.line
        self.advance()  # Skip ⍝

        result = ""
        while self.current_char() and self.current_char() != '\n':
            result += self.advance()

        return Token(TokenType.COMMENT, result.strip(), start_line, start_col)

    def read_system_name(self) -> Token:
        """Read a system variable like ⎕IO"""
        start_col = self.column
        start_line = self.line
        self.advance()  # Skip ⎕

        name = "⎕"
        while self.current_char() and self.current_char() in NAME_CHARS:
            name += self.advance()

        if len(name) > 1:
            return Token(TokenType.SYSTEM_VAR, name, start_line, start_col)
        else:
            return Token(TokenType.QUAD, name, start_line, start_col)

    def tokenize(self) -> List[Token]:
        """Tokenize the entire source"""
        self.tokens = []

        while self.pos < len(self.source):
            char = self.current_char()

            # Skip whitespace (except newlines)
            if char in ' \t':
                self.skip_whitespace()
                continue

            # Newline
            if char == '\n':
                self.add_token(TokenType.NEWLINE, '\n')
                self.advance()
                continue

            # Carriage return (ignore, handle with newline)
            if char == '\r':
                self.advance()
                continue

            # Number (including high-minus)
            if char in '0123456789' or (char == '¯' and self.peek_char() in '0123456789'):
                self.tokens.append(self.read_number())
                continue

            # String
            if char == "'":
                self.tokens.append(self.read_string())
                continue

            # Comment
            if char == '⍝':
                self.tokens.append(self.read_comment())
                continue

            # System variable or quad
            if char == '⎕':
                self.tokens.append(self.read_system_name())
                continue

            # Quote-quad
            if char == '⍞':
                self.add_token(TokenType.QUOTE_QUAD, '⍞')
                self.advance()
                continue

            # Zilde
            if char == '⍬':
                self.add_token(TokenType.ZILDE, '⍬')
                self.advance()
                continue

            # Del
            if char == '∇':
                self.add_token(TokenType.DEL, '∇')
                self.advance()
                continue

            # Alpha
            if char == '⍺':
                self.add_token(TokenType.ALPHA, '⍺')
                self.advance()
                continue

            # Omega
            if char == '⍵':
                self.add_token(TokenType.OMEGA, '⍵')
                self.advance()
                continue

            # Assignment
            if char == '←':
                self.add_token(TokenType.ASSIGN, '←')
                self.advance()
                continue

            # Branch
            if char == '→':
                self.add_token(TokenType.BRANCH, '→')
                self.advance()
                continue

            # Parentheses
            if char == '(':
                self.add_token(TokenType.LPAREN, '(')
                self.advance()
                continue
            if char == ')':
                self.add_token(TokenType.RPAREN, ')')
                self.advance()
                continue

            # Brackets
            if char == '[':
                self.add_token(TokenType.LBRACKET, '[')
                self.advance()
                continue
            if char == ']':
                self.add_token(TokenType.RBRACKET, ']')
                self.advance()
                continue

            # Braces (for dfns)
            if char == '{':
                self.add_token(TokenType.LBRACE, '{')
                self.advance()
                continue
            if char == '}':
                self.add_token(TokenType.RBRACE, '}')
                self.advance()
                continue

            # Semicolon
            if char == ';':
                self.add_token(TokenType.SEMICOLON, ';')
                self.advance()
                continue

            # Colon
            if char == ':':
                self.add_token(TokenType.COLON, ':')
                self.advance()
                continue

            # Diamond (statement separator)
            if char == '⋄':
                self.add_token(TokenType.DIAMOND, '⋄')
                self.advance()
                continue

            # Name/identifier
            if char in NAME_START:
                self.tokens.append(self.read_name())
                continue

            # Operators (check before primitives since some overlap)
            if char in APL_OPERATORS:
                self.add_token(TokenType.OPERATOR, char)
                self.advance()
                continue

            # Primitives
            if char in APL_PRIMITIVES:
                self.add_token(TokenType.PRIMITIVE, char)
                self.advance()
                continue

            # Handle some ASCII that maps to APL
            if char == '+':
                self.add_token(TokenType.PRIMITIVE, '+')
                self.advance()
                continue
            if char == '-':
                self.add_token(TokenType.PRIMITIVE, '-')
                self.advance()
                continue
            if char == '*':
                self.add_token(TokenType.PRIMITIVE, '*')
                self.advance()
                continue
            if char == '/':
                self.add_token(TokenType.OPERATOR, '/')
                self.advance()
                continue
            if char == '\\':
                self.add_token(TokenType.OPERATOR, '\\')
                self.advance()
                continue
            if char == '<':
                self.add_token(TokenType.PRIMITIVE, '<')
                self.advance()
                continue
            if char == '>':
                self.add_token(TokenType.PRIMITIVE, '>')
                self.advance()
                continue
            if char == '=':
                self.add_token(TokenType.PRIMITIVE, '=')
                self.advance()
                continue
            if char == '|':
                self.add_token(TokenType.PRIMITIVE, '|')
                self.advance()
                continue
            if char == ',':
                self.add_token(TokenType.PRIMITIVE, ',')
                self.advance()
                continue
            if char == '~':
                self.add_token(TokenType.PRIMITIVE, '~')
                self.advance()
                continue
            if char == '!':
                self.add_token(TokenType.PRIMITIVE, '!')
                self.advance()
                continue
            if char == '?':
                self.add_token(TokenType.PRIMITIVE, '?')
                self.advance()
                continue

            # Unknown character - skip it
            self.advance()

        self.add_token(TokenType.EOF, None)
        return self.tokens


# =============================================================================
# APL Parser
# =============================================================================

class APLParser:
    """
    Parses APL tokens into an AST.

    APL has unusual evaluation rules:
    - Right-to-left evaluation
    - Functions can be monadic (one arg) or dyadic (two args)
    - Context determines which
    - Operators modify functions
    """

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current_token(self) -> Token:
        """Get current token"""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # EOF

    def peek_token(self, offset: int = 1) -> Token:
        """Look ahead"""
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return self.tokens[-1]

    def advance(self) -> Token:
        """Move to next token and return current"""
        token = self.current_token()
        self.pos += 1
        return token

    def expect(self, token_type: TokenType) -> Token:
        """Expect a specific token type"""
        token = self.current_token()
        if token.type != token_type:
            raise SyntaxError(
                f"Expected {token_type.name}, got {token.type.name} at line {token.line}"
            )
        return self.advance()

    def skip_newlines(self):
        """Skip newline tokens"""
        while self.current_token().type == TokenType.NEWLINE:
            self.advance()

    def is_at_end(self) -> bool:
        """Check if at end of tokens"""
        return self.current_token().type == TokenType.EOF

    def is_function_token(self, token: Token) -> bool:
        """Check if token represents a function"""
        if token.type == TokenType.PRIMITIVE:
            return True
        if token.type == TokenType.NAME:
            # User-defined functions are names
            return True
        return False

    def is_value_start(self, token: Token) -> bool:
        """Check if token can start a value/expression"""
        return token.type in (
            TokenType.NUMBER, TokenType.STRING, TokenType.NAME,
            TokenType.LPAREN, TokenType.LBRACE, TokenType.ZILDE,
            TokenType.QUAD, TokenType.QUOTE_QUAD, TokenType.ALPHA,
            TokenType.OMEGA, TokenType.SYSTEM_VAR
        )

    def parse_program(self) -> Program:
        """Parse a complete APL program (multiple functions)"""
        functions = []
        variables = {}

        while not self.is_at_end():
            self.skip_newlines()
            if self.is_at_end():
                break

            # Check for function definition
            if self.current_token().type == TokenType.DEL:
                func = self.parse_function_def()
                if func:
                    functions.append(func)
            else:
                # Skip unknown content
                self.advance()

        return Program(functions=functions, variables=variables)

    def parse_function_def(self) -> Optional[FunctionDef]:
        """Parse a function definition starting with ∇"""
        if self.current_token().type != TokenType.DEL:
            return None

        self.advance()  # Skip ∇
        self.skip_newlines()

        # Collect header tokens until newline
        header_tokens = []
        while self.current_token().type not in (TokenType.NEWLINE, TokenType.EOF):
            header_tokens.append(self.advance())

        self.skip_newlines()

        # Parse header
        header_info = self._parse_function_header(header_tokens)

        # Collect body lines until closing ∇
        body_tokens: List[List[Token]] = []
        current_line: List[Token] = []
        raw_lines: List[str] = []

        while self.current_token().type != TokenType.EOF:
            token = self.current_token()

            if token.type == TokenType.DEL:
                self.advance()
                break

            if token.type == TokenType.NEWLINE:
                if current_line:
                    body_tokens.append(current_line)
                    raw_lines.append(self._tokens_to_string(current_line))
                    current_line = []
                self.advance()
            else:
                current_line.append(self.advance())

        if current_line:
            body_tokens.append(current_line)
            raw_lines.append(self._tokens_to_string(current_line))

        # Parse body statements
        body = []
        for line_tokens in body_tokens:
            if line_tokens:
                stmt = self._parse_statement(line_tokens)
                if stmt:
                    body.append(stmt)

        header_str = self._tokens_to_string(header_tokens)

        return FunctionDef(
            name=header_info['name'],
            header=header_str,
            result_var=header_info['result'],
            left_arg=header_info['left_arg'],
            right_arg=header_info['right_arg'],
            locals=header_info['locals'],
            body=body,
            raw_lines=raw_lines
        )

    def _parse_function_header(self, tokens: List[Token]) -> Dict[str, Any]:
        """Parse function header like: Z←L NAME R;loc1;loc2"""
        result = {
            'name': 'UNKNOWN',
            'result': None,
            'left_arg': None,
            'right_arg': None,
            'locals': []
        }

        if not tokens:
            return result

        # Split on semicolons for locals
        main_tokens = []
        current_local = []
        in_locals = False

        for token in tokens:
            if token.type == TokenType.SEMICOLON:
                if not in_locals:
                    main_tokens = current_local
                    in_locals = True
                else:
                    if current_local:
                        local_name = self._tokens_to_string(current_local).strip()
                        if local_name:
                            result['locals'].append(local_name)
                current_local = []
            else:
                current_local.append(token)

        if in_locals and current_local:
            local_name = self._tokens_to_string(current_local).strip()
            if local_name:
                result['locals'].append(local_name)
        elif not in_locals:
            main_tokens = current_local

        if not main_tokens:
            return result

        # Check for result variable (assignment)
        assign_idx = None
        for i, token in enumerate(main_tokens):
            if token.type == TokenType.ASSIGN:
                assign_idx = i
                break

        if assign_idx is not None and assign_idx > 0:
            # Result variable is before ←
            result_tokens = main_tokens[:assign_idx]
            result['result'] = self._tokens_to_string(result_tokens).strip()
            main_tokens = main_tokens[assign_idx + 1:]

        # Remaining tokens are: [LEFTARG] NAME [RIGHTARG]
        names = [t for t in main_tokens if t.type == TokenType.NAME]

        if len(names) == 1:
            # Niladic or monadic with single arg
            result['name'] = names[0].value
        elif len(names) == 2:
            # NAME RIGHTARG (monadic)
            result['name'] = names[0].value
            result['right_arg'] = names[1].value
        elif len(names) >= 3:
            # LEFTARG NAME RIGHTARG (dyadic)
            result['left_arg'] = names[0].value
            result['name'] = names[1].value
            result['right_arg'] = names[2].value

        return result

    def _tokens_to_string(self, tokens: List[Token]) -> str:
        """Convert tokens back to string representation"""
        parts = []
        for token in tokens:
            if token.type == TokenType.NUMBER:
                if isinstance(token.value, float) and token.value < 0:
                    parts.append(f"¯{abs(token.value)}")
                elif isinstance(token.value, int) and token.value < 0:
                    parts.append(f"¯{abs(token.value)}")
                else:
                    parts.append(str(token.value))
            elif token.type == TokenType.STRING:
                parts.append(f"'{token.value}'")
            elif token.type == TokenType.COMMENT:
                parts.append(f"⍝ {token.value}")
            else:
                parts.append(str(token.value) if token.value else '')
        return ' '.join(parts)

    def _parse_statement(self, tokens: List[Token]) -> Optional[Statement]:
        """Parse a single statement from its tokens"""
        if not tokens:
            return None

        label = None
        comment = None

        # Check for label at start
        if (len(tokens) >= 2 and
            tokens[0].type == TokenType.NAME and
            tokens[1].type == TokenType.COLON):
            label = tokens[0].value
            tokens = tokens[2:]

        # Check for comment at end
        for i, token in enumerate(tokens):
            if token.type == TokenType.COMMENT:
                comment = token.value
                tokens = tokens[:i]
                break

        if not tokens:
            return Statement(label=label, expr=None, comment=comment)

        # Parse the expression
        parser = APLExpressionParser(tokens)
        expr = parser.parse()

        return Statement(label=label, expr=expr, comment=comment)

    def parse_expression(self) -> Optional[ASTNode]:
        """Parse a single expression (line)"""
        # Collect tokens until newline or EOF
        tokens = []
        while self.current_token().type not in (TokenType.NEWLINE, TokenType.EOF):
            tokens.append(self.advance())

        if self.current_token().type == TokenType.NEWLINE:
            self.advance()

        if not tokens:
            return None

        parser = APLExpressionParser(tokens)
        return parser.parse()


class APLExpressionParser:
    """
    Parses APL expressions using right-to-left evaluation.

    This is the heart of APL parsing - handling the context-sensitive
    determination of monadic vs dyadic function application.
    """

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current_token(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def peek_token(self, offset: int = 1) -> Optional[Token]:
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None

    def advance(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            self.pos += 1
            return token
        return None

    def parse(self) -> Optional[ASTNode]:
        """Parse the expression"""
        if not self.tokens:
            return None

        # Handle comment-only lines
        if len(self.tokens) == 1 and self.tokens[0].type == TokenType.COMMENT:
            return Comment(text=self.tokens[0].value)

        return self.parse_expression()

    def parse_expression(self) -> Optional[ASTNode]:
        """
        Parse an APL expression using right-to-left evaluation.

        APL parsing is tricky because:
        1. Same symbol can be monadic or dyadic depending on context
        2. Evaluation is right-to-left
        3. Operators modify functions

        Strategy: Parse from left, but evaluate right-to-left by
        building the tree appropriately.
        """
        # First, check for assignment
        assign_pos = self._find_assignment()
        if assign_pos is not None:
            return self._parse_assignment(assign_pos)

        # Check for branch
        if self.current_token() and self.current_token().type == TokenType.BRANCH:
            self.advance()
            target = self.parse_expression()
            return Branch(target=target)

        # Parse right-to-left
        return self.parse_right_to_left()

    def _find_assignment(self) -> Optional[int]:
        """Find the leftmost assignment operator"""
        for i, token in enumerate(self.tokens):
            if token.type == TokenType.ASSIGN:
                return i
        return None

    def _parse_assignment(self, assign_pos: int) -> ASTNode:
        """Parse an assignment expression"""
        # Target is everything before ←
        target_tokens = self.tokens[:assign_pos]
        # Value is everything after ←
        value_tokens = self.tokens[assign_pos + 1:]

        # Parse target (usually a name, but could be indexed)
        if len(target_tokens) == 1 and target_tokens[0].type == TokenType.NAME:
            target_name = target_tokens[0].value
            is_indexed = False
            indices = None
        elif len(target_tokens) == 1 and target_tokens[0].type == TokenType.QUAD:
            # ⎕←X is quad output
            value_parser = APLExpressionParser(value_tokens)
            value = value_parser.parse()
            return QuadOutput(value=value)
        else:
            # Could be indexed assignment X[I]←Y
            target_name = target_tokens[0].value if target_tokens else 'UNKNOWN'
            is_indexed = any(t.type == TokenType.LBRACKET for t in target_tokens)
            indices = None  # TODO: parse indices

        # Parse value
        value_parser = APLExpressionParser(value_tokens)
        value = value_parser.parse()

        return Assignment(
            target=target_name,
            value=value,
            is_indexed=is_indexed,
            indices=indices
        )

    def parse_right_to_left(self) -> Optional[ASTNode]:
        """
        Parse using APL's right-to-left evaluation.

        The trick is that we read left-to-right but build the tree
        such that rightmost operations are deepest (evaluated first).
        """
        # Collect all elements
        elements = []

        while self.current_token():
            token = self.current_token()

            # Comment ends expression
            if token.type == TokenType.COMMENT:
                break

            # Parse atomic values and track functions/operators
            if token.type == TokenType.NUMBER:
                elements.append(self._parse_number())
            elif token.type == TokenType.STRING:
                elements.append(StringLiteral(value=self.advance().value))
            elif token.type == TokenType.NAME:
                elements.append(Variable(name=self.advance().value))
            elif token.type == TokenType.LPAREN:
                elements.append(self._parse_paren())
            elif token.type == TokenType.LBRACE:
                elements.append(self._parse_dfn())
            elif token.type == TokenType.ZILDE:
                self.advance()
                elements.append(Zilde())
            elif token.type in (TokenType.QUAD, TokenType.SYSTEM_VAR):
                name = self.advance().value
                elements.append(SystemVariable(name=name))
            elif token.type == TokenType.QUOTE_QUAD:
                self.advance()
                elements.append(QuoteQuadInput())
            elif token.type == TokenType.ALPHA:
                self.advance()
                elements.append(Variable(name='⍺'))
            elif token.type == TokenType.OMEGA:
                self.advance()
                elements.append(Variable(name='⍵'))
            elif token.type == TokenType.PRIMITIVE:
                elements.append(('FUNC', self.advance().value))
            elif token.type == TokenType.OPERATOR:
                elements.append(('OP', self.advance().value))
            elif token.type == TokenType.LBRACKET:
                # Indexing or axis specification
                indices = self._parse_index_spec()
                elements.append(('INDEX', indices))
            else:
                break

        if not elements:
            return None

        # Now evaluate right-to-left
        return self._evaluate_elements(elements)

    def _parse_number(self) -> ASTNode:
        """Parse a number or numeric strand"""
        numbers = [self.advance().value]

        # Check for strand (space-separated numbers)
        while (self.current_token() and
               self.current_token().type == TokenType.NUMBER):
            numbers.append(self.advance().value)

        if len(numbers) == 1:
            return NumberLiteral(value=numbers[0])
        else:
            return NumberLiteral(value=numbers)

    def _parse_paren(self) -> ASTNode:
        """Parse parenthesized expression"""
        self.advance()  # Skip (

        # Collect tokens until matching )
        depth = 1
        inner_tokens = []

        while self.current_token() and depth > 0:
            token = self.current_token()
            if token.type == TokenType.LPAREN:
                depth += 1
            elif token.type == TokenType.RPAREN:
                depth -= 1
                if depth == 0:
                    self.advance()  # Skip )
                    break

            if depth > 0:
                inner_tokens.append(self.advance())

        # Parse inner expression
        inner_parser = APLExpressionParser(inner_tokens)
        inner_expr = inner_parser.parse()

        return ParenExpr(expr=inner_expr) if inner_expr else NumberLiteral(value=0)

    def _parse_dfn(self) -> ASTNode:
        """Parse direct function {body}"""
        self.advance()  # Skip {

        depth = 1
        body_tokens = []

        while self.current_token() and depth > 0:
            token = self.current_token()
            if token.type == TokenType.LBRACE:
                depth += 1
            elif token.type == TokenType.RBRACE:
                depth -= 1
                if depth == 0:
                    self.advance()
                    break

            if depth > 0:
                body_tokens.append(self.advance())

        body_str = ' '.join(str(t.value) for t in body_tokens if t.value)
        return DfnLambda(body=body_str, lines=[body_str])

    def _parse_index_spec(self) -> List[Optional[ASTNode]]:
        """Parse index specification [I] or [I;J;K]"""
        self.advance()  # Skip [

        indices = []
        current_tokens = []

        while self.current_token():
            token = self.current_token()

            if token.type == TokenType.RBRACKET:
                self.advance()
                break
            elif token.type == TokenType.SEMICOLON:
                # Parse current index
                if current_tokens:
                    parser = APLExpressionParser(current_tokens)
                    indices.append(parser.parse())
                else:
                    indices.append(None)  # All elements for this axis
                current_tokens = []
                self.advance()
            else:
                current_tokens.append(self.advance())

        # Handle last index
        if current_tokens:
            parser = APLExpressionParser(current_tokens)
            indices.append(parser.parse())
        elif indices:  # Trailing semicolon
            indices.append(None)

        return indices if indices else [None]

    def _evaluate_elements(self, elements: List) -> Optional[ASTNode]:
        """
        Evaluate a list of elements right-to-left.

        Elements can be:
        - ASTNode (values)
        - ('FUNC', symbol) (functions)
        - ('OP', symbol) (operators)
        - ('INDEX', indices) (indexing)
        """
        if not elements:
            return None

        # Work from right to left
        elements = list(elements)  # Copy

        # Start with rightmost value
        result = None
        i = len(elements) - 1

        while i >= 0:
            elem = elements[i]

            if isinstance(elem, ASTNode):
                if result is None:
                    result = elem
                else:
                    # Two values in a row - this is stranding
                    if isinstance(result, Strand):
                        result.elements.insert(0, elem)
                    elif isinstance(elem, Strand):
                        elem.elements.append(result)
                        result = elem
                    else:
                        result = Strand(elements=[elem, result])

            elif isinstance(elem, tuple):
                tag, value = elem

                if tag == 'INDEX':
                    # Apply indexing to result
                    if result:
                        result = IndexExpr(array=result, indices=value)

                elif tag == 'OP':
                    # Operator - look for function to its left
                    if i > 0:
                        prev = elements[i - 1]
                        if isinstance(prev, tuple) and prev[0] == 'FUNC':
                            # f/ pattern - reduction or replication
                            func_symbol = prev[1]
                            i -= 1
                            if result:
                                result = OperatorExpr(
                                    operator=value,
                                    func=func_symbol,
                                    operand=result
                                )
                        else:
                            # Operator without function - might be scan on implicit +
                            if result:
                                result = OperatorExpr(
                                    operator=value,
                                    func='+',  # Default
                                    operand=result
                                )
                    else:
                        # Standalone operator
                        if result:
                            result = OperatorExpr(
                                operator=value,
                                func='+',
                                operand=result
                            )

                elif tag == 'FUNC':
                    # Function - check if monadic or dyadic
                    if i > 0:
                        prev = elements[i - 1]
                        if isinstance(prev, ASTNode):
                            # Have left arg - dyadic
                            i -= 1
                            if result:
                                result = DyadicCall(
                                    func=value,
                                    left=prev,
                                    right=result
                                )
                            else:
                                result = prev
                        else:
                            # No left arg - monadic
                            if result:
                                result = MonadicCall(func=value, arg=result)
                    else:
                        # Leftmost - monadic
                        if result:
                            result = MonadicCall(func=value, arg=result)

            i -= 1

        return result


# =============================================================================
# Python Code Generator
# =============================================================================

class PythonGenerator:
    """
    Generates Python code from APL AST.

    Uses NumPy for array operations and the apl_primitives library
    for APL-specific functions.
    """

    def __init__(self):
        self.indent_level = 0
        self.indent_str = "    "
        self.imports = {
            'numpy': 'np',
            'apl_primitives': '*'
        }
        self.generated_functions: List[str] = []

        # APL to Python identifier mapping
        self.apl_to_python_names = {
            '∆': 'Delta',
            '⍙': 'DeltaBar',
            '⍺': 'alpha',
            '⍵': 'omega',
            '⎕': 'Quad',
            '⍞': 'QuoteQuad',
            '⍬': 'zilde',
            '⊂': 'enclose',
            '⊃': 'disclose',
            '⊖': 'rotate_first',
            '⍉': 'transpose',
            '⍕': 'format',
            '⍎': 'execute',
            '⍴': 'rho',
            '⍳': 'iota',
            '○': 'circle',
            '⌈': 'ceiling',
            '⌊': 'floor',
            '⍋': 'grade_up',
            '⍒': 'grade_down',
            '⌽': 'reverse',
            '∊': 'epsilon',
            '⍷': 'find',
            '⌿': 'reduce_first',
            '⍀': 'scan_first',
        }

    def _sanitize_identifier(self, name: str) -> str:
        """
        Convert APL identifier to valid Python identifier.

        Handles APL special characters and ensures Python naming rules are followed.
        """
        if not name:
            return 'unknown_var'

        # Replace APL special characters with Python-friendly names
        sanitized = name
        for apl_char, py_name in self.apl_to_python_names.items():
            sanitized = sanitized.replace(apl_char, py_name)

        # Remove any remaining invalid characters
        result = []
        for char in sanitized:
            if char.isalnum() or char == '_':
                result.append(char)
            else:
                # Replace invalid char with underscore
                result.append('_')

        sanitized = ''.join(result)

        # Ensure starts with letter or underscore
        if sanitized and not (sanitized[0].isalpha() or sanitized[0] == '_'):
            sanitized = 'var_' + sanitized

        # Handle Python keywords
        python_keywords = {
            'if', 'else', 'elif', 'for', 'while', 'def', 'class', 'return',
            'import', 'from', 'as', 'with', 'try', 'except', 'finally',
            'raise', 'assert', 'break', 'continue', 'pass', 'lambda',
            'yield', 'global', 'nonlocal', 'del', 'is', 'in', 'not',
            'and', 'or', 'True', 'False', 'None'
        }
        if sanitized.lower() in python_keywords:
            sanitized = sanitized + '_'

        # Handle empty result
        if not sanitized or sanitized == '_':
            return 'unknown_var'

        return sanitized

    def indent(self) -> str:
        """Get current indentation"""
        return self.indent_str * self.indent_level

    def generate_program(self, program: Program) -> str:
        """Generate Python code for entire program"""
        lines = []

        # Header comment
        lines.append('"""')
        lines.append("Transpiled from APL")
        lines.append("")
        lines.append("Generated by APL to Python Transpiler")
        lines.append('"""')
        lines.append("")

        # Imports
        lines.append("import numpy as np")
        lines.append("from apl_primitives import *")
        lines.append("")
        lines.append("")

        # Generate each function
        for func in program.functions:
            func_code = self.generate_function(func)
            lines.append(func_code)
            lines.append("")
            lines.append("")

        return '\n'.join(lines)

    def generate_function(self, func: FunctionDef) -> str:
        """Generate Python function from APL function definition"""
        lines = []

        # Sanitize function name
        clean_func_name = self._sanitize_identifier(func.name)

        # Build parameter list with sanitized names
        params = []
        if func.left_arg:
            params.append(self._sanitize_identifier(func.left_arg))
        if func.right_arg:
            params.append(self._sanitize_identifier(func.right_arg))

        param_str = ', '.join(params)

        # Function signature
        lines.append(f"def {clean_func_name}({param_str}):")

        self.indent_level = 1

        # Docstring
        lines.append(f'{self.indent()}"""')
        lines.append(f'{self.indent()}Transpiled from APL function: {func.name}')
        if func.header:
            lines.append(f'{self.indent()}Original header: {func.header}')
        lines.append(f'{self.indent()}"""')

        # Local variables initialization (with sanitized names)
        for local in func.locals:
            clean_local = self._sanitize_identifier(local)
            lines.append(f"{self.indent()}{clean_local} = None")

        if func.locals:
            lines.append("")

        # Track labels for goto handling
        labels = self._collect_labels(func.body)
        has_branches = self._has_branches(func.body)

        if has_branches and labels:
            # Use state machine for control flow
            lines.extend(self._generate_state_machine(func, labels))
        else:
            # Simple sequential code
            for stmt in func.body:
                stmt_code = self.generate_statement(stmt)
                if stmt_code:
                    lines.append(f"{self.indent()}{stmt_code}")

        # Return statement (with sanitized name)
        if func.result_var:
            clean_result = self._sanitize_identifier(func.result_var)
            lines.append(f"{self.indent()}return {clean_result}")

        self.indent_level = 0
        return '\n'.join(lines)

    def _collect_labels(self, body: List[ASTNode]) -> Dict[str, int]:
        """Collect all labels and their statement indices"""
        labels = {}
        for i, stmt in enumerate(body):
            if isinstance(stmt, Statement) and stmt.label:
                labels[stmt.label] = i
        return labels

    def _has_branches(self, body: List[ASTNode]) -> bool:
        """Check if function has any branch statements"""
        for stmt in body:
            if isinstance(stmt, Statement) and isinstance(stmt.expr, Branch):
                return True
        return False

    def _generate_state_machine(self, func: FunctionDef, labels: Dict[str, int]) -> List[str]:
        """Generate state machine code for functions with branches"""
        lines = []

        lines.append(f"{self.indent()}_state = 0")
        lines.append(f"{self.indent()}_labels = {labels}")
        lines.append("")
        lines.append(f"{self.indent()}while True:")

        self.indent_level += 1

        for i, stmt in enumerate(func.body):
            if isinstance(stmt, Statement) and stmt.label:
                lines.append(f"{self.indent()}# {stmt.label}:")

            lines.append(f"{self.indent()}if _state == {i}:")
            self.indent_level += 1

            if isinstance(stmt, Statement):
                if isinstance(stmt.expr, Branch):
                    # Branch handling
                    target = self.generate_expression(stmt.expr.target)
                    lines.append(f"{self.indent()}_target = {target}")
                    lines.append(f"{self.indent()}if _target == 0:")
                    lines.append(f"{self.indent()}    break  # Exit function")
                    lines.append(f"{self.indent()}elif isinstance(_target, str) and _target in _labels:")
                    lines.append(f"{self.indent()}    _state = _labels[_target]")
                    lines.append(f"{self.indent()}    continue")
                    lines.append(f"{self.indent()}elif isinstance(_target, (int, float)):")
                    lines.append(f"{self.indent()}    _state = int(_target) - 1")
                    lines.append(f"{self.indent()}    continue")
                else:
                    stmt_code = self.generate_statement(stmt)
                    if stmt_code:
                        lines.append(f"{self.indent()}{stmt_code}")

                lines.append(f"{self.indent()}_state = {i + 1}")

            self.indent_level -= 1

        # End condition
        lines.append(f"{self.indent()}if _state >= {len(func.body)}:")
        self.indent_level += 1
        lines.append(f"{self.indent()}break")
        self.indent_level -= 1

        self.indent_level -= 1

        return lines

    def generate_statement(self, stmt: Statement) -> Optional[str]:
        """Generate Python code for a statement"""
        if not stmt.expr:
            if stmt.comment:
                return f"# {stmt.comment}"
            return None

        code = self.generate_expression(stmt.expr)

        if stmt.comment:
            code = f"{code}  # {stmt.comment}"

        return code

    def generate_expression(self, node: ASTNode) -> str:
        """Generate Python code for an expression"""
        if node is None:
            return "None"

        if isinstance(node, NumberLiteral):
            if isinstance(node.value, list):
                return f"np.array({node.value})"
            return str(node.value)

        if isinstance(node, StringLiteral):
            return repr(node.value)

        if isinstance(node, Variable):
            name = node.name
            # Handle APL special names
            if name == '⍺':
                return '_alpha'
            if name == '⍵':
                return '_omega'
            # Sanitize all variable names
            return self._sanitize_identifier(name)

        if isinstance(node, SystemVariable):
            name = node.name
            if name == '⎕IO':
                return 'INDEX_ORIGIN'
            return f"quad_{name[1:].lower()}"

        if isinstance(node, Zilde):
            return "zilde()"

        if isinstance(node, Assignment):
            value = self.generate_expression(node.value)
            clean_target = self._sanitize_identifier(node.target)
            return f"{clean_target} = {value}"

        if isinstance(node, MonadicCall):
            return self._generate_monadic(node.func, node.arg)

        if isinstance(node, DyadicCall):
            return self._generate_dyadic(node.func, node.left, node.right)

        if isinstance(node, OperatorExpr):
            return self._generate_operator(node)

        if isinstance(node, IndexExpr):
            return self._generate_index(node)

        if isinstance(node, Branch):
            target = self.generate_expression(node.target)
            return f"_branch({target})"

        if isinstance(node, QuadOutput):
            value = self.generate_expression(node.value)
            return f"quad_output({value})"

        if isinstance(node, QuoteQuadInput):
            return "quote_quad_input()"

        if isinstance(node, Strand):
            elements = [self.generate_expression(e) for e in node.elements]
            return f"np.array([{', '.join(elements)}])"

        if isinstance(node, ParenExpr):
            inner = self.generate_expression(node.expr)
            return f"({inner})"

        if isinstance(node, DfnLambda):
            return self._generate_dfn(node)

        if isinstance(node, Comment):
            return f"# {node.text}"

        # Fallback
        return f"# TODO: {type(node).__name__}"

    def _generate_monadic(self, func: str, arg: ASTNode) -> str:
        """Generate code for monadic function call"""
        arg_code = self.generate_expression(arg)

        # Map APL primitives to Python/NumPy
        mappings = {
            '+': f"conjugate({arg_code})",
            '-': f"negate({arg_code})",
            '×': f"signum({arg_code})",
            '÷': f"reciprocal({arg_code})",
            '|': f"magnitude({arg_code})",
            '⌊': f"floor({arg_code})",
            '⌈': f"ceiling({arg_code})",
            '*': f"exponential({arg_code})",
            '⍟': f"natural_log({arg_code})",
            '○': f"pi_times({arg_code})",
            '!': f"factorial({arg_code})",
            '?': f"roll({arg_code})",
            '~': f"logical_not({arg_code})",
            '⍳': f"iota({arg_code})",
            '⍴': f"shape({arg_code})",
            ',': f"ravel({arg_code})",
            '⌽': f"reverse({arg_code})",
            '⊖': f"reverse_first({arg_code})",
            '⍉': f"transpose({arg_code})",
            '⍋': f"grade_up({arg_code})",
            '⍒': f"grade_down({arg_code})",
            '∪': f"unique({arg_code})",
            '⊂': f"enclose({arg_code})",
            '⊃': f"disclose({arg_code})",
            '↑': f"first({arg_code})",
            '≡': f"depth({arg_code})",
            '≢': f"tally({arg_code})",
            '⌹': f"matrix_inverse({arg_code})",
            '⍕': f"format_array({arg_code})",
            '⍎': f"execute({arg_code})",
            '⊢': f"same({arg_code})",
        }

        if func in mappings:
            return mappings[func]

        # User-defined function
        return f"{func}({arg_code})"

    def _generate_dyadic(self, func: str, left: ASTNode, right: ASTNode) -> str:
        """Generate code for dyadic function call"""
        left_code = self.generate_expression(left)
        right_code = self.generate_expression(right)

        # Map APL primitives to Python/NumPy
        mappings = {
            '+': f"plus({left_code}, {right_code})",
            '-': f"minus({left_code}, {right_code})",
            '×': f"times({left_code}, {right_code})",
            '÷': f"divide({left_code}, {right_code})",
            '|': f"residue({left_code}, {right_code})",
            '⌊': f"minimum({left_code}, {right_code})",
            '⌈': f"maximum({left_code}, {right_code})",
            '*': f"power({left_code}, {right_code})",
            '⍟': f"logarithm({left_code}, {right_code})",
            '○': f"circular({left_code}, {right_code})",
            '!': f"binomial({left_code}, {right_code})",
            '?': f"deal({left_code}, {right_code})",
            '=': f"equal({left_code}, {right_code})",
            '≠': f"not_equal({left_code}, {right_code})",
            '<': f"less_than({left_code}, {right_code})",
            '≤': f"less_equal({left_code}, {right_code})",
            '>': f"greater_than({left_code}, {right_code})",
            '≥': f"greater_equal({left_code}, {right_code})",
            '∧': f"logical_and({left_code}, {right_code})",
            '∨': f"logical_or({left_code}, {right_code})",
            '⍲': f"logical_nand({left_code}, {right_code})",
            '⍱': f"logical_nor({left_code}, {right_code})",
            '⍴': f"rho({left_code}, {right_code})",
            ',': f"catenate({left_code}, {right_code})",
            '⍪': f"catenate_first({left_code}, {right_code})",
            '↑': f"take({left_code}, {right_code})",
            '↓': f"drop({left_code}, {right_code})",
            '⌽': f"rotate({left_code}, {right_code})",
            '⊖': f"rotate_first({left_code}, {right_code})",
            '⍉': f"dyadic_transpose({left_code}, {right_code})",
            '⍳': f"index_of({left_code}, {right_code})",
            '∊': f"membership({left_code}, {right_code})",
            '∩': f"intersection({left_code}, {right_code})",
            '∪': f"union({left_code}, {right_code})",
            '~': f"without({left_code}, {right_code})",
            '⊤': f"encode({left_code}, {right_code})",
            '⊥': f"decode({left_code}, {right_code})",
            '⌹': f"matrix_divide({left_code}, {right_code})",
            '⊣': f"left_tack({left_code}, {right_code})",
            '⊢': f"right_tack({left_code}, {right_code})",
        }

        if func in mappings:
            return mappings[func]

        # User-defined function
        return f"{func}({left_code}, {right_code})"

    def _generate_operator(self, node: OperatorExpr) -> str:
        """Generate code for operator expression"""
        operand_code = self.generate_expression(node.operand)
        func = node.func if isinstance(node.func, str) else 'plus'

        if node.operator == '/':
            # Reduce
            reduce_funcs = {
                '+': f"reduce_plus({operand_code})",
                '-': f"reduce_minus({operand_code})",
                '×': f"reduce_times({operand_code})",
                '÷': f"reduce_divide({operand_code})",
                '⌈': f"reduce_max({operand_code})",
                '⌊': f"reduce_min({operand_code})",
                '∧': f"reduce_and({operand_code})",
                '∨': f"reduce_or({operand_code})",
            }
            if func in reduce_funcs:
                return reduce_funcs[func]
            return f"reduce_{func}({operand_code})"

        elif node.operator == '\\':
            # Scan
            scan_funcs = {
                '+': f"scan_plus({operand_code})",
                '×': f"scan_times({operand_code})",
                '⌈': f"scan_max({operand_code})",
                '⌊': f"scan_min({operand_code})",
            }
            if func in scan_funcs:
                return scan_funcs[func]
            return f"scan_{func}({operand_code})"

        elif node.operator == '⌿':
            # Reduce first
            return f"reduce_plus({operand_code}, axis=0)"

        elif node.operator == '⍀':
            # Scan first
            return f"scan_plus({operand_code}, axis=0)"

        elif node.operator == '¨':
            # Each
            return f"np.vectorize(lambda x: {func}(x))({operand_code})"

        elif node.operator == '⍨':
            # Commute/selfie
            return f"{func}({operand_code}, {operand_code})"

        # Default
        return f"# TODO: operator {node.operator} with {func}"

    def _generate_index(self, node: IndexExpr) -> str:
        """Generate code for index expression"""
        array_code = self.generate_expression(node.array)

        index_parts = []
        for idx in node.indices:
            if idx is None:
                index_parts.append(':')
            else:
                idx_code = self.generate_expression(idx)
                # APL uses 1-based indexing by default
                index_parts.append(f"{idx_code} - INDEX_ORIGIN")

        if len(index_parts) == 1:
            return f"{array_code}[{index_parts[0]}]"
        else:
            return f"{array_code}[{', '.join(index_parts)}]"

    def _generate_dfn(self, node: DfnLambda) -> str:
        """Generate code for direct function"""
        # Simple dfn becomes lambda
        body = node.body
        # Replace APL special vars
        body = body.replace('⍺', '_alpha').replace('⍵', '_omega')
        return f"lambda _alpha, _omega: {body}"


# =============================================================================
# APL Transpiler - Main Orchestrator
# =============================================================================

class APLTranspiler:
    """
    Main transpiler class that orchestrates the full pipeline.

    Can work with:
    - ANS files (via ANSDecompiler)
    - APL source files
    - Direct APL strings
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.lexer = None
        self.parser = None
        self.generator = PythonGenerator()

    def log(self, msg: str):
        """Print verbose message"""
        if self.verbose:
            print(f"[TRANSPILER] {msg}")

    def transpile_string(self, apl_source: str) -> str:
        """Transpile APL source code string to Python"""
        self.log("Lexing APL source...")
        self.lexer = APLLexer(apl_source)
        tokens = self.lexer.tokenize()
        self.log(f"Generated {len(tokens)} tokens")

        self.log("Parsing tokens...")
        self.parser = APLParser(tokens)
        program = self.parser.parse_program()
        self.log(f"Parsed {len(program.functions)} functions")

        self.log("Generating Python code...")
        python_code = self.generator.generate_program(program)

        return python_code

    def transpile_file(self, filepath: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Transpile an APL or ANS file to Python.

        Args:
            filepath: Path to .apl or .ans file
            output_dir: Output directory (default: ./output)

        Returns:
            Dictionary with results
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        output_path = Path(output_dir) if output_dir else Path("./output")

        if filepath.suffix.lower() == '.ans':
            return self._transpile_ans(filepath, output_path)
        else:
            return self._transpile_apl(filepath, output_path)

    def _transpile_ans(self, filepath: Path, output_dir: Path) -> Dict[str, Any]:
        """Transpile ANS file via decompiler"""
        # Import the decompiler
        try:
            from ans_decompiler import ANSDecompiler
        except ImportError:
            raise ImportError("ans_decompiler.py not found")

        self.log(f"Decompiling ANS file: {filepath}")

        # Decompile first
        decompiler = ANSDecompiler(str(filepath), verbose=self.verbose)
        functions, variables = decompiler.decompile()

        ws_name = decompiler.header.workspace_name if decompiler.header else "WORKSPACE"
        ws_name = ws_name.replace(' ', '_').replace('/', '_')

        # Export APL source
        ws_dir = output_dir / ws_name
        decompiler.export_text(output_dir)

        # Build APL source from functions
        apl_lines = []
        for func in functions:
            apl_lines.append(f"∇ {func.header.replace('∇', '').strip()}")
            for line in func.lines:
                apl_lines.append(f"  {line}")
            apl_lines.append("∇")
            apl_lines.append("")

        apl_source = '\n'.join(apl_lines)

        # Transpile to Python
        python_code = self.transpile_string(apl_source)

        # Write Python output
        python_dir = ws_dir / "python"
        python_dir.mkdir(parents=True, exist_ok=True)

        # Write main module
        main_file = python_dir / "workspace.py"
        main_file.write_text(python_code, encoding='utf-8')

        # Write __init__.py
        init_file = python_dir / "__init__.py"
        init_file.write_text(
            f'"""Transpiled APL workspace: {ws_name}"""\nfrom .workspace import *\n',
            encoding='utf-8'
        )

        # Copy primitives library
        primitives_src = Path(__file__).parent / "apl_primitives.py"
        if primitives_src.exists():
            primitives_dst = python_dir / "apl_primitives.py"
            primitives_dst.write_text(primitives_src.read_text(), encoding='utf-8')

        return {
            "workspace_name": ws_name,
            "apl_functions": len(functions),
            "output_dir": str(ws_dir),
            "python_file": str(main_file),
            "success": True
        }

    def _transpile_apl(self, filepath: Path, output_dir: Path) -> Dict[str, Any]:
        """Transpile APL source file"""
        self.log(f"Reading APL file: {filepath}")

        apl_source = filepath.read_text(encoding='utf-8')

        # Transpile
        python_code = self.transpile_string(apl_source)

        # Determine output location
        name = filepath.stem
        ws_dir = output_dir / name
        python_dir = ws_dir / "python"
        python_dir.mkdir(parents=True, exist_ok=True)

        # Write output
        main_file = python_dir / f"{name}.py"
        main_file.write_text(python_code, encoding='utf-8')

        return {
            "source_file": str(filepath),
            "output_dir": str(ws_dir),
            "python_file": str(main_file),
            "success": True
        }


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main entry point for CLI usage"""
    parser = argparse.ArgumentParser(
        description="Transpile APL/ANS files to Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s SQL.ans                    # Transpile ANS file
  %(prog)s workspace.apl              # Transpile APL source
  %(prog)s SQL.ans -o mydir           # Custom output directory
  %(prog)s SQL.ans --python-only      # Skip APL output (ANS only)
  %(prog)s SQL.ans -v                 # Verbose output
        """
    )

    parser.add_argument(
        "files",
        nargs="*",
        help="APL or ANS file(s) to transpile"
    )

    parser.add_argument(
        "-o", "--output",
        default="./output",
        help="Output directory (default: ./output)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose debug information"
    )

    parser.add_argument(
        "--python-only",
        action="store_true",
        help="Only generate Python output (skip APL for ANS files)"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick test with sample APL code"
    )

    args = parser.parse_args()

    # Run test mode
    if args.test:
        run_test()
        return 0

    # Validate files are provided
    if not args.files:
        parser.error("the following arguments are required: files (use --test for demo)")

    # Process files
    transpiler = APLTranspiler(verbose=args.verbose)
    results = []

    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}", file=sys.stderr)
            continue

        print(f"Transpiling: {filepath}")

        try:
            result = transpiler.transpile_file(filepath, args.output)
            results.append(result)

            print(f"  Output: {result.get('python_file', result.get('output_dir'))}")

        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            results.append({
                "file": filepath,
                "success": False,
                "error": str(e)
            })

    # Summary
    print()
    successful = sum(1 for r in results if r.get("success"))
    print(f"Processed {len(results)} file(s), {successful} successful")

    return 0 if successful == len(results) else 1


def run_test():
    """Run a quick test of the transpiler"""
    print("Running transpiler test...")
    print()

    # Test APL code
    apl_code = """
∇ Z←AVERAGE X;S;N
  S←+/X
  N←⍴X
  Z←S÷N
∇

∇ Z←FACTORIAL N
  →(N≤1)/L1
  Z←N×FACTORIAL N-1
  →0
L1:Z←1
∇

∇ Z←A PLUS B
  Z←A+B
∇
"""

    print("Input APL:")
    print("-" * 40)
    print(apl_code)
    print("-" * 40)
    print()

    transpiler = APLTranspiler(verbose=True)
    python_code = transpiler.transpile_string(apl_code)

    print()
    print("Generated Python:")
    print("-" * 40)
    print(python_code)
    print("-" * 40)

    return 0


if __name__ == "__main__":
    sys.exit(main())
