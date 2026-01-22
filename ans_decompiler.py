#!/usr/bin/env python3
"""
APL2 ANS File Decompiler

Parses and decompiles IBM APL2 .ans (APL Namespace Save) workspace files,
extracting functions, variables, and other objects into readable format.

Usage:
    python ans_decompiler.py SQL.ans              # Decompile to ./output/SQL/
    python ans_decompiler.py SQL.ans -o mydir     # Custom output directory
    python ans_decompiler.py SQL.ans --transpile  # Also generate Python code
    python ans_decompiler.py *.ans                # Batch process multiple files
"""

import argparse
import json
import os
import struct
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# IBM APL2 to Unicode translation table
# Maps IBM APL2 character set byte values to Unicode APL symbols
APL2_TO_UNICODE = {
    # Control characters and basic ASCII (0x00-0x1F)
    0x00: '\x00',  # Null
    0x01: '\x01',
    0x02: '\x02',
    0x03: '\x03',
    0x04: '\x04',
    0x05: '\x05',
    0x06: '\x06',
    0x07: '\x07',
    0x08: '\x08',
    0x09: '\t',    # Tab
    0x0A: '\n',    # Newline
    0x0B: '\x0b',
    0x0C: '\x0c',
    0x0D: '\r',    # Carriage return
    0x0E: '\x0e',
    0x0F: '\x0f',
    0x10: '\x10',
    0x11: '\x11',
    0x12: '\x12',
    0x13: '\x13',
    0x14: '\x14',
    0x15: '\x15',
    0x16: '\x16',
    0x17: '\x17',
    0x18: '\x18',
    0x19: '\x19',
    0x1A: '\x1a',
    0x1B: '\x1b',
    0x1C: '\x1c',
    0x1D: '\x1d',
    0x1E: '\x1e',
    0x1F: '\x1f',

    # Standard ASCII printable (0x20-0x7E)
    0x20: ' ',
    0x21: '!',
    0x22: '"',
    0x23: '#',
    0x24: '$',
    0x25: '%',
    0x26: '&',
    0x27: "'",
    0x28: '(',
    0x29: ')',
    0x2A: '*',
    0x2B: '+',
    0x2C: ',',
    0x2D: '-',
    0x2E: '.',
    0x2F: '/',
    0x30: '0',
    0x31: '1',
    0x32: '2',
    0x33: '3',
    0x34: '4',
    0x35: '5',
    0x36: '6',
    0x37: '7',
    0x38: '8',
    0x39: '9',
    0x3A: ':',
    0x3B: ';',
    0x3C: '<',
    0x3D: '=',
    0x3E: '>',
    0x3F: '?',
    0x40: '@',
    0x41: 'A',
    0x42: 'B',
    0x43: 'C',
    0x44: 'D',
    0x45: 'E',
    0x46: 'F',
    0x47: 'G',
    0x48: 'H',
    0x49: 'I',
    0x4A: 'J',
    0x4B: 'K',
    0x4C: 'L',
    0x4D: 'M',
    0x4E: 'N',
    0x4F: 'O',
    0x50: 'P',
    0x51: 'Q',
    0x52: 'R',
    0x53: 'S',
    0x54: 'T',
    0x55: 'U',
    0x56: 'V',
    0x57: 'W',
    0x58: 'X',
    0x59: 'Y',
    0x5A: 'Z',
    0x5B: '[',
    0x5C: '\\',
    0x5D: ']',
    0x5E: '^',
    0x5F: '_',
    0x60: '`',
    0x61: 'a',
    0x62: 'b',
    0x63: 'c',
    0x64: 'd',
    0x65: 'e',
    0x66: 'f',
    0x67: 'g',
    0x68: 'h',
    0x69: 'i',
    0x6A: 'j',
    0x6B: 'k',
    0x6C: 'l',
    0x6D: 'm',
    0x6E: 'n',
    0x6F: 'o',
    0x70: 'p',
    0x71: 'q',
    0x72: 'r',
    0x73: 's',
    0x74: 't',
    0x75: 'u',
    0x76: 'v',
    0x77: 'w',
    0x78: 'x',
    0x79: 'y',
    0x7A: 'z',
    0x7B: '{',
    0x7C: '|',
    0x7D: '}',
    0x7E: '~',
    0x7F: '\x7f',

    # APL special characters (0x80-0xFF)
    # These mappings are based on IBM APL2 character set
    0x80: '⌶',  # I-beam
    0x81: '¨',  # Dieresis
    0x82: '¯',  # Macron (high minus)
    0x83: '<',  # Less than
    0x84: '≤',  # Less than or equal
    0x85: '=',  # Equal
    0x86: '≥',  # Greater than or equal
    0x87: '>',  # Greater than
    0x88: '≠',  # Not equal
    0x89: '∨',  # Or (logical)
    0x8A: '∧',  # And (logical)
    0x8B: '×',  # Times (multiply)
    0x8C: '÷',  # Divide
    0x8D: '?',  # Question mark / Roll
    0x8E: '⍵',  # Omega
    0x8F: '∊',  # Epsilon (membership)
    0x90: '⍴',  # Rho (shape)
    0x91: '~',  # Tilde (not)
    0x92: '↑',  # Up arrow (take/mix)
    0x93: '↓',  # Down arrow (drop/split)
    0x94: '⍳',  # Iota (index generator)
    0x95: '○',  # Circle (pi times/trig)
    0x96: '*',  # Star (power/exponential)
    0x97: '⌈',  # Ceiling
    0x98: '⌊',  # Floor
    0x99: '∇',  # Del (function definition)
    0x9A: '∆',  # Delta
    0x9B: '⊤',  # Encode (up tack)
    0x9C: '⊥',  # Decode (down tack)
    0x9D: '|',  # Stile (absolute/residue)
    0x9E: ',',  # Comma (catenate/ravel)
    0x9F: '⍎',  # Execute
    0xA0: '⍕',  # Format
    0xA1: '⊂',  # Left shoe (enclose)
    0xA2: '⊃',  # Right shoe (disclose/pick)
    0xA3: '∩',  # Intersection
    0xA4: '∪',  # Union
    0xA5: '⊣',  # Left tack
    0xA6: '⊢',  # Right tack
    0xA7: '⌷',  # Squad (index)
    0xA8: '⍋',  # Grade up
    0xA9: '⍒',  # Grade down
    0xAA: '⍫',  # Del stile
    0xAB: '⍱',  # Nor
    0xAC: '⍲',  # Nand
    0xAD: '⍟',  # Log
    0xAE: '⌹',  # Domino (matrix divide/inverse)
    0xAF: '⌽',  # Circle stile (rotate/reverse)
    0xB0: '⊖',  # Circle bar (rotate first/reverse first)
    0xB1: '⍉',  # Transpose
    0xB2: '!',  # Factorial/binomial
    0xB3: '⌿',  # Slash bar (reduce first)
    0xB4: '⍀',  # Backslash bar (expand first)
    0xB5: '/',  # Slash (reduce/replicate)
    0xB6: '\\', # Backslash (scan/expand)
    0xB7: '⍝',  # Lamp (comment)
    0xB8: '⍲',  # Nand (alternate)
    0xB9: '⍱',  # Nor (alternate)
    0xBA: '¤',  # Currency symbol
    0xBB: '⌻',  # Quad divide
    0xBC: '⌺',  # Quad diamond
    0xBD: '⌾',  # Circle jot
    0xBE: '⍃',  # Less than underbar
    0xBF: '⍄',  # Greater than underbar
    0xC0: '←',  # Left arrow (assignment)
    0xC1: '→',  # Right arrow (branch/goto)
    0xC2: '⍬',  # Zilde (empty numeric vector)
    0xC3: '⍺',  # Alpha
    0xC4: '⌈',  # Ceiling (alternate)
    0xC5: '⌊',  # Floor (alternate)
    0xC6: '_',  # Underscore
    0xC7: '∇',  # Del (alternate)
    0xC8: '∘',  # Jot (compose/null)
    0xC9: '\'', # Quote
    0xCA: '⎕',  # Quad
    0xCB: '⍞',  # Quote quad
    0xCC: '¢',  # Cent sign
    0xCD: '⊆',  # Left shoe underbar
    0xCE: '⊇',  # Right shoe underbar
    0xCF: '⍸',  # Iota underbar
    0xD0: '⍷',  # Epsilon underbar (find)
    0xD1: '⌸',  # Quad equal (key)
    0xD2: '⍤',  # Jot dieresis (rank)
    0xD3: '⍥',  # Circle dieresis
    0xD4: '⍣',  # Star dieresis (power operator)
    0xD5: '⍨',  # Tilde dieresis (commute)
    0xD6: '⍠',  # Quad colon (variant)
    0xD7: '⌸',  # Key
    0xD8: '⌼',  # Circle stile
    0xD9: '⍁',  # Quad slash
    0xDA: '⍂',  # Quad backslash
    0xDB: '⍃',  # Less underbar
    0xDC: '⍄',  # Greater underbar
    0xDD: '⍅',  # Left vane
    0xDE: '⍆',  # Right vane
    0xDF: '⍏',  # Up vane
    0xE0: '⍖',  # Down vane
    0xE1: '⍊',  # Up tack jot
    0xE2: '⍑',  # Down tack jot
    0xE3: '⍘',  # Underbar dieresis
    0xE4: '⍙',  # Delta underbar
    0xE5: '⍚',  # Diamond underbar
    0xE6: '⍛',  # Jot underbar
    0xE7: '⍜',  # Circle underbar
    0xE8: '⍮',  # Semicolon underbar
    0xE9: '⍡',  # Up tack overbar
    0xEA: '⍢',  # Del dieresis
    0xEB: '⍩',  # Greater dieresis
    0xEC: '⍦',  # Down shoe stile
    0xED: '⍧',  # Left shoe stile
    0xEE: '⍪',  # Comma bar (table)
    0xEF: '⍫',  # Del tilde
    0xF0: '⍭',  # Stile tilde
    0xF1: '⍮',  # Semicolon underbar (alternate)
    0xF2: '⍯',  # Not equal underbar
    0xF3: '⍰',  # Quad question
    0xF4: '‿',  # Tie (undertie)
    0xF5: '¶',  # Pilcrow / paragraph
    0xF6: '§',  # Section
    0xF7: '·',  # Middle dot
    0xF8: '¸',  # Cedilla
    0xF9: '°',  # Degree
    0xFA: '±',  # Plus-minus
    0xFB: '²',  # Superscript 2
    0xFC: '³',  # Superscript 3
    0xFD: '´',  # Acute accent
    0xFE: 'µ',  # Micro sign
    0xFF: '¿',  # Inverted question mark
}

# Alternative EBCDIC-based APL2 character set mapping
# Used when the file appears to use EBCDIC encoding
APL2_EBCDIC_TO_UNICODE = {
    # Common EBCDIC APL mappings
    0x40: ' ',   # Space
    0x4A: '¢',   # Cent
    0x4B: '.',   # Period
    0x4C: '<',   # Less than
    0x4D: '(',   # Left paren
    0x4E: '+',   # Plus
    0x4F: '|',   # Vertical bar
    0x50: '&',   # Ampersand
    0x5A: '!',   # Exclamation
    0x5B: '$',   # Dollar
    0x5C: '*',   # Asterisk
    0x5D: ')',   # Right paren
    0x5E: ';',   # Semicolon
    0x5F: '¬',   # Not
    0x60: '-',   # Minus
    0x61: '/',   # Slash
    0x6A: '¦',   # Broken bar
    0x6B: ',',   # Comma
    0x6C: '%',   # Percent
    0x6D: '_',   # Underscore
    0x6E: '>',   # Greater than
    0x6F: '?',   # Question
    0x79: '`',   # Grave accent
    0x7A: ':',   # Colon
    0x7B: '#',   # Hash
    0x7C: '@',   # At
    0x7D: "'",   # Apostrophe
    0x7E: '=',   # Equals
    0x7F: '"',   # Quote

    # Uppercase letters
    0xC1: 'A', 0xC2: 'B', 0xC3: 'C', 0xC4: 'D', 0xC5: 'E',
    0xC6: 'F', 0xC7: 'G', 0xC8: 'H', 0xC9: 'I',
    0xD1: 'J', 0xD2: 'K', 0xD3: 'L', 0xD4: 'M', 0xD5: 'N',
    0xD6: 'O', 0xD7: 'P', 0xD8: 'Q', 0xD9: 'R',
    0xE2: 'S', 0xE3: 'T', 0xE4: 'U', 0xE5: 'V', 0xE6: 'W',
    0xE7: 'X', 0xE8: 'Y', 0xE9: 'Z',

    # Lowercase letters
    0x81: 'a', 0x82: 'b', 0x83: 'c', 0x84: 'd', 0x85: 'e',
    0x86: 'f', 0x87: 'g', 0x88: 'h', 0x89: 'i',
    0x91: 'j', 0x92: 'k', 0x93: 'l', 0x94: 'm', 0x95: 'n',
    0x96: 'o', 0x97: 'p', 0x98: 'q', 0x99: 'r',
    0xA2: 's', 0xA3: 't', 0xA4: 'u', 0xA5: 'v', 0xA6: 'w',
    0xA7: 'x', 0xA8: 'y', 0xA9: 'z',

    # Numbers
    0xF0: '0', 0xF1: '1', 0xF2: '2', 0xF3: '3', 0xF4: '4',
    0xF5: '5', 0xF6: '6', 0xF7: '7', 0xF8: '8', 0xF9: '9',

    # APL special characters in EBCDIC
    0x70: '⍺',   # Alpha
    0x71: '⊥',   # Up tack (decode)
    0x72: '⌈',   # Ceiling
    0x73: '⌊',   # Floor
    0x74: '_',   # Underscore
    0x75: '∇',   # Del
    0x76: '∆',   # Delta
    0x77: '⍳',   # Iota
    0x78: '∘',   # Jot
    0x8A: '[',   # Left bracket
    0x8B: ']',   # Right bracket
    0x9A: '⊤',   # Down tack (encode)
    0x9B: '⍵',   # Omega
    0xA0: '⍬',   # Zilde
    0xA1: '~',   # Tilde
    0xAA: '{',   # Left brace
    0xAB: '}',   # Right brace
    0xAD: '⍝',   # Comment lamp
    0xB0: '⍴',   # Rho
    0xB1: '←',   # Left arrow
    0xB2: '→',   # Right arrow
    0xB3: '⌹',   # Domino
    0xB4: '⎕',   # Quad
    0xB5: '⍞',   # Quote quad
    0xB6: '\\',  # Backslash
    0xB7: '÷',   # Divide
    0xBA: '×',   # Times
    0xBB: '↑',   # Up arrow
    0xBC: '↓',   # Down arrow
    0xBD: '⊂',   # Enclose
    0xBE: '⊃',   # Disclose
    0xBF: '⌷',   # Squad
    0xCA: '¨',   # Dieresis
    0xCB: '⍎',   # Execute
    0xCC: '⍕',   # Format
    0xCD: '⍋',   # Grade up
    0xCE: '⍒',   # Grade down
    0xCF: '○',   # Circle
    0xDA: '⌽',   # Reverse
    0xDB: '⊖',   # Rotate first
    0xDC: '⍉',   # Transpose
    0xDD: '⍟',   # Log
    0xDE: '∊',   # Epsilon
    0xDF: '⍷',   # Find
    0xEA: '∨',   # Or
    0xEB: '∧',   # And
    0xEC: '≠',   # Not equal
    0xED: '≤',   # Less equal
    0xEE: '≥',   # Greater equal
    0xEF: '¯',   # High minus
    0xFA: '⌿',   # Reduce first
    0xFB: '⍀',   # Expand first
    0xFC: '∩',   # Intersection
    0xFD: '∪',   # Union
    0xFE: '⊣',   # Left tack
    0xFF: '⊢',   # Right tack
}


# =============================================================================
# APL2 Bytecode to APL Symbol Mapping
# =============================================================================
# APL2 stores code as tokenized bytecode, not plain text.
# This maps bytecode values to their APL symbols.

APL2_BYTECODE_TO_SYMBOL = {
    # Structural/Control
    0x00B8: '→',      # Branch (goto)
    0x00BD: '←',      # Assignment
    0x00C0: '←',      # Assignment (alternate encoding)
    0x00C1: '→',      # Branch (alternate)

    # Parentheses and brackets
    0x0A28: '(',      # Left paren
    0x0928: '(',      # Left paren (alternate)
    0x0028: '(',      # Left paren (ASCII)
    0x0A29: ')',      # Right paren
    0x0929: ')',      # Right paren (alternate)
    0x0029: ')',      # Right paren (ASCII)
    0x005B: '[',      # Left bracket
    0x005D: ']',      # Right bracket
    0x007B: '{',      # Left brace
    0x007D: '}',      # Right brace

    # Arithmetic primitives
    0x002B: '+',      # Plus
    0x002D: '-',      # Minus
    0x002A: '*',      # Times/Power (ASCII asterisk)
    0x008B: '×',      # Times
    0x002F: '/',      # Divide/Reduce
    0x008C: '÷',      # Divide
    0x007C: '|',      # Magnitude/Residue
    0x009D: '|',      # Stile (alternate)

    # Comparison
    0x003D: '=',      # Equal
    0x003C: '<',      # Less than
    0x003E: '>',      # Greater than
    0x0084: '≤',      # Less or equal
    0x0086: '≥',      # Greater or equal
    0x0088: '≠',      # Not equal

    # Logical
    0x0089: '∨',      # Or
    0x008A: '∧',      # And
    0x0091: '~',      # Not/Without
    0x00AB: '⍱',      # Nor
    0x00AC: '⍲',      # Nand

    # Structural primitives
    0x0090: '⍴',      # Rho (shape/reshape)
    0x0094: '⍳',      # Iota (index generator/index of)
    0x009E: ',',      # Comma (catenate/ravel)
    0x00EE: '⍪',      # Comma bar (table/catenate first)
    0x0092: '↑',      # Take/Mix
    0x0093: '↓',      # Drop/Split
    0x00A1: '⊂',      # Enclose
    0x00A2: '⊃',      # Disclose/Pick
    0x00A3: '∩',      # Intersection
    0x00A4: '∪',      # Union
    0x00A7: '⌷',      # Squad (index)

    # Sorting/Ordering
    0x00A8: '⍋',      # Grade up
    0x00A9: '⍒',      # Grade down

    # Rotation/Transposition
    0x00AF: '⌽',      # Reverse/Rotate
    0x00B0: '⊖',      # Reverse first/Rotate first
    0x00B1: '⍉',      # Transpose

    # Math functions
    0x0097: '⌈',      # Ceiling/Maximum
    0x0098: '⌊',      # Floor/Minimum
    0x0095: '○',      # Circle (pi times/trig)
    0x0096: '*',      # Star (power/exponential)
    0x00AD: '⍟',      # Log
    0x00B2: '!',      # Factorial/Binomial
    0x008D: '?',      # Roll/Deal
    0x00AE: '⌹',      # Matrix divide/inverse

    # Operators
    0x012F: '/',      # Reduce operator
    0x00B5: '/',      # Slash (reduce/replicate)
    0x00B6: '\\',     # Backslash (scan/expand)
    0x0B3:  '⌿',      # Slash bar (reduce first)
    0x00B4: '⍀',      # Backslash bar (expand first)
    0x0081: '¨',      # Each (dieresis)
    0x00D5: '⍨',      # Commute (tilde dieresis)
    0x00D4: '⍣',      # Power operator
    0x00C8: '∘',      # Jot (compose)
    0x00D2: '⍤',      # Rank operator

    # I/O and system
    0x00CA: '⎕',      # Quad
    0x00CB: '⍞',      # Quote quad
    0x009F: '⍎',      # Execute
    0x00A0: '⍕',      # Format
    0x00B7: '⍝',      # Comment lamp

    # Special values
    0x00C2: '⍬',      # Zilde (empty numeric)
    0x008F: '∊',      # Epsilon (membership/enlist)
    0x00D0: '⍷',      # Find
    0x0082: '¯',      # High minus

    # Greek letters (used in dfns)
    0x00C3: '⍺',      # Alpha
    0x008E: '⍵',      # Omega

    # Delimiters
    0x0099: '∇',      # Del (function definition)
    0x009A: '∆',      # Delta
    0x003B: ';',      # Semicolon
    0x003A: ':',      # Colon

    # Statement separator
    0x22C4: '⋄',      # Diamond

    # Tacks
    0x00A5: '⊣',      # Left tack
    0x00A6: '⊢',      # Right tack

    # Comparison underbar variants
    0x009B: '⊤',      # Encode
    0x009C: '⊥',      # Decode

    # =========================================================================
    # EXPANDED BYTECODE MAPPINGS - Critical missing patterns
    # =========================================================================

    # System variables (⎕ + 2-byte codes)
    0xCA49: '⎕IO',    # Index origin
    0xCA4F: '⎕IO',    # Index origin (alternate)
    0xCA50: '⎕PP',    # Print precision
    0xCA43: '⎕CT',    # Comparison tolerance
    0xCA4C: '⎕LX',    # Latent expression
    0xCA52: '⎕RL',    # Random link
    0xCA57: '⎕WA',    # Workspace available
    0xCA41: '⎕A',     # Alphabet
    0xCA44: '⎕D',     # Digits
    0xCA41: '⎕AV',    # Atomic vector
    0xCA54: '⎕T',     # Time
    0xCA53: '⎕S',     # State indicator
    0xCA4E: '⎕NL',    # Name list
    0xCA4D: '⎕M',     # Multiplication
    0xCA45: '⎕E',     # Event
    0xCA46: '⎕F',     # Format control
    0xCA47: '⎕G',     # Graphics
    0xCA48: '⎕H',     # Hold
    0xCA4A: '⎕J',     # Job
    0xCA4B: '⎕K',     # Key
    0xCA51: '⎕Q',     # Quote

    # Data type prefixes (markers for decoder state changes)
    0x00FC: '<NUM>',     # Numeric literal follows (4-8 bytes)
    0x00FE: '<STR>',     # String follows (length + data)
    0x00FA: '<NAMEREF>', # Name reference
    0x00FB: '<NAMEDEF>', # Name definition
    0x00FD: '<EMPTY>',   # Empty/null marker

    # Extended operators (multi-byte sequences)
    0xD000: '⍷',  # Find (alternate encoding)
    0xD100: '⌸',  # Key (alternate encoding)
    0xD200: '⍤',  # Rank (alternate encoding)
    0xD300: '⍥',  # Circle dieresis (alternate)
    0xD400: '⍣',  # Power operator (alternate encoding)
    0xD500: '⍨',  # Commute (alternate encoding)
    0xD600: '⍠',  # Variant (alternate encoding)

    # Assignment variants (multiple encodings)
    0x00B1: '←',  # EBCDIC left arrow
    0x10C0: '←',  # Left arrow with prefix
    0x10C1: '→',  # Right arrow with prefix

    # Additional comparison operators
    0x1084: '≤',  # Less or equal (with prefix)
    0x1086: '≥',  # Greater or equal (with prefix)
    0x1088: '≠',  # Not equal (with prefix)

    # Array operators
    0x00CD: '⊆',  # Left shoe underbar (partitioned enclose)
    0x00CE: '⊇',  # Right shoe underbar (partitioned pick)
    0x00CF: '⍸',  # Iota underbar (where/interval index)

    # Advanced operators
    0x00D1: '⌸',  # Quad equal (key operator)
    0x00D3: '⍥',  # Circle dieresis (over/atop)
    0x00D6: '⍠',  # Quad colon (variant operator)

    # Additional structural primitives
    0x10A1: '⊂',  # Enclose (with prefix)
    0x10A2: '⊃',  # Disclose (with prefix)
    0x10A7: '⌷',  # Squad (with prefix)

    # Additional mathematical functions
    0x1095: '○',  # Circle (with prefix)
    0x1096: '*',  # Star (with prefix)
    0x10AD: '⍟',  # Log (with prefix)
    0x10AE: '⌹',  # Domino (with prefix)

    # Reduce/scan variants with axis
    0x10B3: '⌿',  # Slash bar - reduce first (with prefix)
    0x10B4: '⍀',  # Backslash bar - expand first (with prefix)
    0x10B5: '/',  # Slash (with prefix)
    0x10B6: '\\', # Backslash (with prefix)

    # Control structures
    0x003A: ':',      # Colon (control structure)
    0x22C4: '⋄',      # Diamond (statement separator)
    0x10C4: '⋄',      # Diamond (alternate)

    # Namespace/scope operators
    0x002E: '.',      # Dot (namespace reference)
    0x00E4: '⍙',      # Delta underbar (alternate delta)

    # Additional logical operators
    0x10AB: '⍱',      # Nor (with prefix)
    0x10AC: '⍲',      # Nand (with prefix)

    # Additional rotation/transpose
    0x10AF: '⌽',      # Reverse (with prefix)
    0x10B0: '⊖',      # Reverse first (with prefix)
    0x10B1: '⍉',      # Transpose (with prefix)

    # Additional indexing
    0x10A8: '⍋',      # Grade up (with prefix)
    0x10A9: '⍒',      # Grade down (with prefix)

    # Numeric constants and special values
    0x10C2: '⍬',      # Zilde (with prefix)
    0x1082: '¯',      # High minus (with prefix)

    # String/character operations
    0x1027: "'",      # Quote (with prefix)
    0x1022: '"',      # Double quote (with prefix)

    # Additional set operations
    0x10A3: '∩',      # Intersection (with prefix)
    0x10A4: '∪',      # Union (with prefix)
    0x108F: '∊',      # Epsilon (with prefix)

    # Additional encode/decode
    0x109B: '⊤',      # Encode (with prefix)
    0x109C: '⊥',      # Decode (with prefix)

    # Tack variants
    0x10A5: '⊣',      # Left tack (with prefix)
    0x10A6: '⊢',      # Right tack (with prefix)

    # Special characters for debugging
    0x00: '<NUL>',    # Null byte marker
    0x0A: '\n',       # Newline
    0x0D: '\r',       # Carriage return

    # Additional variants found in IBM APL2
    0x20C0: '←',      # Assignment (double-byte encoding)
    0x20C1: '→',      # Branch (double-byte encoding)

    # Comment and string markers
    0x10B7: '⍝',      # Comment (with prefix)

    # More system functions
    0xCA58: '⎕X',     # Extended precision
    0xCA59: '⎕Y',     # Y variable
    0xCA5A: '⎕Z',     # Z variable

    # Additional APL2 specific operators
    0x0080: '⌶',      # I-beam (alternate)
    0x1080: '⌶',      # I-beam (with prefix)

    # Bracket variants
    0x105B: '[',      # Left bracket (with prefix)
    0x105D: ']',      # Right bracket (with prefix)
    0x107B: '{',      # Left brace (with prefix)
    0x107D: '}',      # Right brace (with prefix)

    # Parenthesis variants
    0x1028: '(',      # Left paren (with prefix)
    0x1029: ')',      # Right paren (with prefix)

    # Additional arithmetic
    0x102B: '+',      # Plus (with prefix)
    0x102D: '-',      # Minus (with prefix)
    0x102A: '*',      # Times (with prefix)
    0x102F: '/',      # Divide (with prefix)

    # Comparison with prefix
    0x103D: '=',      # Equal (with prefix)
    0x103C: '<',      # Less than (with prefix)
    0x103E: '>',      # Greater than (with prefix)

    # Additional special APL characters
    0x00AA: '⍫',      # Del stile (alternate)
    0x00E5: '⍚',      # Diamond underbar
    0x00E6: '⍛',      # Jot underbar
    0x00E7: '⍜',      # Circle underbar
    0x00E8: '⍮',      # Semicolon underbar
    0x00E9: '⍡',      # Up tack overbar
    0x00EA: '⍢',      # Del dieresis
    0x00EB: '⍩',      # Greater dieresis
    0x00EC: '⍦',      # Down shoe stile
    0x00ED: '⍧',      # Left shoe stile
    0x00EF: '⍫',      # Del tilde (alternate)
    0x00F0: '⍭',      # Stile tilde
    0x00F1: '⍮',      # Semicolon underbar (alternate)
    0x00F2: '⍯',      # Not equal underbar
    0x00F3: '⍰',      # Quad question

    # Whitespace variants
    0x1020: ' ',      # Space (with prefix)
    0x1009: '\t',     # Tab (with prefix)
}

# Reverse mapping for encoding
SYMBOL_TO_BYTECODE = {v: k for k, v in APL2_BYTECODE_TO_SYMBOL.items()}


class ANSBinaryParser:
    """
    Parses ANS binary file structure to extract objects directory and data blocks.

    Separates the low-level binary parsing from bytecode decoding to prevent
    corruption and improve accuracy.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def log(self, msg: str):
        """Print debug message"""
        if self.verbose:
            print(f"[BINARY_PARSER] {msg}")

    def parse(self, data: bytes) -> Dict[str, Any]:
        """
        Parse ANS file binary structure.

        Args:
            data: Raw ANS file bytes

        Returns:
            Dictionary containing:
                - magic: File magic string
                - version: Version string
                - workspace_name: Name of workspace
                - object_count: Number of objects
                - objects: List of object dictionaries with offset, size, type, name, data
        """
        if len(data) < 0x100:
            raise ValueError("File too small to be valid ANS file")

        result = {
            'magic': '',
            'version': '',
            'workspace_name': '',
            'object_count': 0,
            'objects': []
        }

        # Parse magic number at offset 0x00 (typically "ANS 1.00" or similar)
        magic_raw = data[0:16]
        result['magic'] = magic_raw.rstrip(b'\x00').decode('ascii', errors='replace').strip()
        self.log(f"Magic: {result['magic']}")

        # Extract version (often embedded in magic or nearby)
        if b'1.00' in magic_raw or b'1.00' in data[0:32]:
            result['version'] = "1.00"

        # Workspace name at offset 0x30-0x3F (16 bytes)
        ws_name_raw = data[0x30:0x40]
        result['workspace_name'] = ws_name_raw.rstrip(b'\x00').decode('ascii', errors='replace').strip()
        self.log(f"Workspace: {result['workspace_name']}")

        # Object count at offset 0x2C (4 bytes, little-endian)
        if len(data) >= 0x30:
            obj_count_le = struct.unpack('<I', data[0x2C:0x30])[0]
            obj_count_be = struct.unpack('>I', data[0x2C:0x30])[0]

            # Use the more reasonable count
            if obj_count_le < 10000:
                result['object_count'] = obj_count_le
            elif obj_count_be < 10000:
                result['object_count'] = obj_count_be
            else:
                result['object_count'] = 0

            self.log(f"Object count: {result['object_count']}")

        # Find object directory
        # The directory typically starts at a fixed offset or can be found by pattern
        dir_offset = self._find_object_directory(data)
        self.log(f"Object directory found at: 0x{dir_offset:X}")

        # Parse object directory entries
        result['objects'] = self._parse_object_directory(data, dir_offset, result['object_count'])
        self.log(f"Parsed {len(result['objects'])} objects")

        return result

    def _find_object_directory(self, data: bytes) -> int:
        """
        Locate the object directory in the file.

        Object directories typically have patterns like:
        - Sequential offset values (increasing)
        - Readable names (uppercase letters)
        - Fixed-size entries (often 32 bytes each)

        Returns:
            Offset to start of object directory
        """
        # Common object directory locations in ANS files
        potential_offsets = [
            0x100,  # After header (256 bytes)
            0x200,  # After extended header
            0x45F0, # Observed in some files
            0x4000,
            0x3000,
            0x2000,
            0x1000,
        ]

        for offset in potential_offsets:
            if offset >= len(data) - 32:
                continue

            # Check if this looks like an object directory
            # Look for patterns indicating structured data
            chunk = data[offset:offset + 256]

            # Count readable ASCII uppercase letters (common in object names)
            readable_upper = sum(1 for b in chunk if 0x41 <= b <= 0x5A)
            readable_alpha = sum(1 for b in chunk if 0x41 <= b <= 0x5A or 0x61 <= b <= 0x7A)

            # Object directories typically have many null bytes (padding in names)
            null_bytes = sum(1 for b in chunk if b == 0x00)

            # Good indicators: lots of uppercase letters, many nulls (padding)
            if readable_upper > 15 and null_bytes > 50:
                self.log(f"Potential directory at 0x{offset:X} (upper={readable_upper}, nulls={null_bytes})")
                return offset

        # Default: search for function markers (∇ symbols) and work backwards
        return self._search_for_directory_by_functions(data)

    def _search_for_directory_by_functions(self, data: bytes) -> int:
        """Find directory by locating functions and working backwards"""
        # Look for del markers (∇) which indicate functions
        del_markers = [0x99, 0xC7, 0x75]  # Various encodings of ∇

        first_function_offset = None
        for i in range(0x100, min(len(data), 0x10000)):
            if data[i] in del_markers:
                first_function_offset = i
                break

        if first_function_offset:
            # Directory is likely before first function
            # Round down to nearest 256-byte boundary
            dir_offset = (first_function_offset // 256) * 256
            self.log(f"Directory inferred from function at 0x{first_function_offset:X}")
            return max(0x100, dir_offset - 0x100)

        # Fallback
        return 0x100

    def _parse_object_directory(self, data: bytes, dir_offset: int, expected_count: int) -> List[Dict[str, Any]]:
        """
        Parse object directory entries.

        Each entry typically contains:
        - Offset (4 bytes)
        - Size (4 bytes)
        - Type (1 byte): 1=function, 2=variable, etc.
        - Flags (1 byte)
        - Padding (2 bytes)
        - Name (16 bytes, null-padded)

        Total: 32 bytes per entry (may vary by format version)
        """
        objects = []
        entry_size = 32
        max_entries = min(expected_count if expected_count > 0 else 1000, 1000)

        for i in range(max_entries):
            entry_offset = dir_offset + (i * entry_size)

            if entry_offset + entry_size > len(data):
                break

            entry_data = data[entry_offset:entry_offset + entry_size]

            # Try to parse entry
            try:
                # Offset (4 bytes, little-endian)
                obj_offset = struct.unpack('<I', entry_data[0:4])[0]

                # Size (4 bytes, little-endian)
                obj_size = struct.unpack('<I', entry_data[4:8])[0]

                # Type (1 byte)
                obj_type = entry_data[8]

                # Flags (1 byte)
                obj_flags = entry_data[9]

                # Name (remaining bytes, typically at offset 16)
                obj_name_raw = entry_data[16:32]
                obj_name = obj_name_raw.rstrip(b'\x00').decode('ascii', errors='replace').strip()

                # Validate entry - must have reasonable values
                if obj_offset > 0 and obj_offset < len(data) and obj_size > 0 and obj_size < len(data):
                    # Extract actual data
                    obj_data = data[obj_offset:obj_offset + obj_size] if obj_offset + obj_size <= len(data) else b''

                    # Only add if name looks valid (has at least one letter)
                    if obj_name and any(c.isalpha() for c in obj_name):
                        objects.append({
                            'offset': obj_offset,
                            'size': obj_size,
                            'type': obj_type,
                            'flags': obj_flags,
                            'name': obj_name,
                            'data': obj_data
                        })
                        self.log(f"  Object: {obj_name} at 0x{obj_offset:X}, size={obj_size}, type={obj_type}")
                else:
                    # Invalid entry - might have reached end of directory
                    if len(objects) > 0:
                        break

            except (struct.error, IndexError, UnicodeDecodeError):
                # Malformed entry - might be end of directory
                if len(objects) > 0:
                    break

        return objects


class BytecodeDecoder:
    """
    Decodes APL2 tokenized bytecode into APL source text.

    APL2 stores functions and expressions as bytecode tokens rather than
    plain text. This decoder interprets those bytecodes and reconstructs
    readable APL source code.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def log(self, msg: str):
        """Print debug message"""
        if self.verbose:
            print(f"[BYTECODE] {msg}")

    def decode_with_boundaries(self, data: bytes, boundaries: Dict[str, int]) -> Tuple[str, List[str]]:
        """
        Decode function with known boundaries using state-aware logic.

        Args:
            data: Raw bytecode bytes
            boundaries: Dict with 'start', 'end', 'header_end' keys

        Returns:
            Tuple of (header_line, body_lines)
        """
        if not boundaries or 'header_end' not in boundaries:
            # Fallback to regular decoding
            return self.decode_function(data)

        start = boundaries.get('start', 0)
        header_end = boundaries['header_end']
        end = boundaries.get('end', len(data))

        # Decode header (from start to header_end)
        header_data = data[start:header_end]
        header = self.decode(header_data).strip()
        header = header.replace('∇', '').strip()

        # Decode body (from header_end to end)
        body_data = data[header_end:end]
        body_text = self.decode(body_data)

        # Split into lines and clean up
        body_lines = [l.rstrip() for l in body_text.split('\n') if l.strip()]

        # Remove closing del if present
        if body_lines and body_lines[-1].strip() == '∇':
            body_lines = body_lines[:-1]

        return header, body_lines

    def decode_numeric(self, data: bytes, pos: int) -> Tuple[Union[int, float], int]:
        """
        Decode numeric literal starting at position.

        Returns:
            Tuple of (number, bytes_consumed)
        """
        # Check for marker
        if pos >= len(data):
            return 0, 0

        # Common numeric encodings in APL2:
        # - 4-byte integers (little-endian)
        # - 8-byte floats (IEEE 754 double)

        bytes_available = len(data) - pos

        # Try to read as double (8 bytes)
        if bytes_available >= 8:
            try:
                value = struct.unpack('<d', data[pos:pos+8])[0]
                # Check if it's a reasonable number
                if not (value != value):  # not NaN
                    return value, 8
            except struct.error:
                pass

        # Try to read as integer (4 bytes)
        if bytes_available >= 4:
            try:
                value = struct.unpack('<i', data[pos:pos+4])[0]
                return value, 4
            except struct.error:
                pass

        # Could not decode
        return 0, 0

    def decode_string(self, data: bytes, pos: int) -> Tuple[str, int]:
        """
        Decode string literal starting at position.

        Returns:
            Tuple of (string, bytes_consumed)
        """
        if pos >= len(data):
            return "", 0

        # Check for quote marker
        if data[pos] == 0x27:  # Single quote
            start = pos + 1
            i = start
            chars = []

            while i < len(data):
                if data[i] == 0x27:  # Closing quote
                    # Check for escaped quote
                    if i + 1 < len(data) and data[i + 1] == 0x27:
                        chars.append("'")
                        i += 2
                    else:
                        # End of string
                        return ''.join(chars), i + 1 - pos
                elif 0x20 <= data[i] <= 0x7E:
                    chars.append(chr(data[i]))
                    i += 1
                else:
                    i += 1

            # Unclosed string
            return ''.join(chars), i - pos

        # No quote marker - try length-prefixed
        if pos + 1 < len(data):
            length = data[pos]
            if length > 0 and length < 200 and pos + 1 + length <= len(data):
                string_data = data[pos + 1:pos + 1 + length]
                try:
                    text = string_data.decode('ascii', errors='replace')
                    return text, 1 + length
                except:
                    pass

        return "", 0

    def decode_name(self, data: bytes, pos: int) -> Tuple[str, int]:
        """
        Decode variable/function name starting at position.

        Returns:
            Tuple of (name, bytes_consumed)
        """
        if pos >= len(data):
            return "", 0

        chars = []
        i = pos

        # Names are alphanumeric plus underscore, delta
        name_bytes = set(range(ord('A'), ord('Z') + 1)) | \
                     set(range(ord('a'), ord('z') + 1)) | \
                     set(range(ord('0'), ord('9') + 1)) | \
                     {ord('_'), 0x9A, 0xE4}  # underscore, delta, delta-underbar

        # First character must be letter or underscore/delta
        if data[i] not in name_bytes or (data[i] >= ord('0') and data[i] <= ord('9')):
            return "", 0

        while i < len(data) and data[i] in name_bytes:
            if data[i] == 0x9A:
                chars.append('∆')
            elif data[i] == 0xE4:
                chars.append('⍙')
            elif 0x20 <= data[i] <= 0x7E:
                chars.append(chr(data[i]))
            i += 1

        return ''.join(chars), i - pos

    def decode(self, data: bytes) -> str:
        """
        Decode bytecode to APL source text.

        Args:
            data: Raw bytecode bytes

        Returns:
            Decoded APL source string
        """
        result = []
        i = 0

        while i < len(data):
            # Try 2-byte sequences first (for extended bytecodes)
            if i + 1 < len(data):
                two_byte = (data[i] << 8) | data[i + 1]
                if two_byte in APL2_BYTECODE_TO_SYMBOL:
                    result.append(APL2_BYTECODE_TO_SYMBOL[two_byte])
                    i += 2
                    continue

            # Try single byte
            byte = data[i]

            # Check bytecode mapping
            if byte in APL2_BYTECODE_TO_SYMBOL:
                result.append(APL2_BYTECODE_TO_SYMBOL[byte])
                i += 1
                continue

            # Standard ASCII printable
            if 0x20 <= byte <= 0x7E:
                result.append(chr(byte))
                i += 1
                continue

            # Newlines and tabs
            if byte == 0x0A:
                result.append('\n')
                i += 1
                continue
            if byte == 0x0D:
                i += 1
                continue
            if byte == 0x09:
                result.append('\t')
                i += 1
                continue

            # Check APL character translation
            if byte in APL2_TO_UNICODE:
                result.append(APL2_TO_UNICODE[byte])
                i += 1
                continue

            # Skip null and other control chars
            if byte < 0x20:
                i += 1
                continue

            # Unknown - include as hex for debugging
            # result.append(f'\\x{byte:02X}')
            i += 1

        return ''.join(result)

    def decode_function(self, data: bytes) -> Tuple[str, List[str]]:
        """
        Decode function bytecode, extracting header and body lines.

        Returns:
            Tuple of (header_line, body_lines)
        """
        decoded = self.decode(data)

        # Split into lines
        lines = decoded.split('\n')
        lines = [l.rstrip() for l in lines if l.strip()]

        if not lines:
            return "", []

        # First line is typically the header
        header = lines[0]

        # Remove del markers from header
        header = header.replace('∇', '').strip()

        # Rest is body
        body = lines[1:]

        # Remove closing del if present
        if body and body[-1].strip() == '∇':
            body = body[:-1]

        return header, body

    def extract_strings(self, data: bytes) -> List[str]:
        """
        Extract embedded string literals from bytecode.

        String literals are often stored as length-prefixed ASCII/EBCDIC
        sequences within the bytecode.
        """
        strings = []
        i = 0

        while i < len(data) - 1:
            # Look for string markers (often 0x27 = single quote)
            if data[i] == 0x27:  # Single quote
                # Read until closing quote
                j = i + 1
                string_chars = []
                while j < len(data):
                    if data[j] == 0x27:
                        # Check for doubled quote (escaped)
                        if j + 1 < len(data) and data[j + 1] == 0x27:
                            string_chars.append("'")
                            j += 2
                        else:
                            break
                    else:
                        if 0x20 <= data[j] <= 0x7E:
                            string_chars.append(chr(data[j]))
                        j += 1
                if string_chars:
                    strings.append(''.join(string_chars))
                i = j + 1
            else:
                i += 1

        return strings

    def extract_names(self, data: bytes) -> List[str]:
        """
        Extract variable and function names from bytecode.

        Names are typically sequences of uppercase letters, digits, and
        delta/underscore characters.
        """
        names = []
        i = 0
        current_name = []

        NAME_CHARS = set(b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_')

        while i < len(data):
            byte = data[i]

            if byte in NAME_CHARS:
                current_name.append(chr(byte))
            else:
                if current_name:
                    name = ''.join(current_name)
                    # Filter out common non-names
                    if (len(name) >= 2 and
                        name[0].isalpha() and
                        name not in ('IF', 'IN', 'OR', 'TO', 'DO')):
                        if name not in names:
                            names.append(name)
                    current_name = []
            i += 1

        return names


class FunctionBoundaryDetector:
    """
    Detects and properly pairs function boundary markers (∇) in ANS bytecode.

    Fixes the issue where functions are incorrectly merged or boundaries are
    misidentified by using proper pairing logic and validation.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        # Del (∇) markers in various encodings
        self.del_markers = {0x99, 0xC7, 0x75}

    def log(self, msg: str):
        """Print debug message"""
        if self.verbose:
            print(f"[BOUNDARY] {msg}")

    def find_functions(self, data: bytes) -> List[Dict[str, int]]:
        """
        Find all function boundaries in bytecode.

        Returns:
            List of dictionaries with 'start', 'end', 'header_end' positions
        """
        # Step 1: Find all del positions
        del_positions = self._find_all_dels(data)
        self.log(f"Found {len(del_positions)} del markers")

        if len(del_positions) < 2:
            return []

        # Step 2: Pair opening and closing dels
        functions = []
        i = 0
        while i < len(del_positions) - 1:
            start_pos = del_positions[i]

            # Find matching closing del
            close_pos = self._find_matching_close(data, start_pos, del_positions[i+1:])

            if close_pos is not None:
                # Find header end (first newline after opening del)
                header_end = self._find_first_newline_after(data, start_pos)

                functions.append({
                    'start': start_pos,
                    'end': close_pos,
                    'header_end': header_end
                })

                self.log(f"Function: 0x{start_pos:X} to 0x{close_pos:X}")

                # Move to after closing del
                try:
                    i = del_positions.index(close_pos) + 1
                except ValueError:
                    i += 1
            else:
                i += 1

        return functions

    def _find_all_dels(self, data: bytes) -> List[int]:
        """Find all del marker positions in data"""
        positions = []
        for i in range(len(data)):
            if data[i] in self.del_markers:
                positions.append(i)
        return positions

    def _find_matching_close(self, data: bytes, start: int, remaining_dels: List[int]) -> Optional[int]:
        """
        Find the closing del that matches the opening del at start.

        A valid closing del must:
        - Be at least 20 bytes after start (minimum function size)
        - Be preceded by a newline (at line start)
        - Be followed by a newline or EOF
        """
        for pos in remaining_dels:
            # Must be reasonable distance from start
            if pos - start < 20:
                continue

            # Check if this del is at start of line
            if not self._is_at_line_start(data, pos):
                continue

            # Check if this is truly a closing del (followed by newline or end)
            if self._is_valid_closing(data, pos):
                return pos

        return None

    def _is_at_line_start(self, data: bytes, pos: int) -> bool:
        """Check if position is at start of a line"""
        if pos == 0:
            return True

        # Look backwards for newline
        # Should be: \n ∇ or start-of-file ∇
        if pos > 0 and data[pos - 1] == 0x0A:  # newline
            return True

        # Also check for \r\n pattern
        if pos > 1 and data[pos - 2] == 0x0D and data[pos - 1] == 0x0A:
            return True

        return False

    def _is_valid_closing(self, data: bytes, pos: int) -> bool:
        """Check if del at pos is a valid closing marker"""
        # Must be preceded by newline (already checked in _is_at_line_start)
        # Must be followed by newline, whitespace, or EOF

        if pos + 1 >= len(data):
            return True  # EOF

        next_byte = data[pos + 1]

        # Followed by newline, space, or null
        if next_byte in {0x0A, 0x0D, 0x20, 0x00}:
            return True

        return False

    def _find_first_newline_after(self, data: bytes, start: int) -> int:
        """Find first newline after start position"""
        for i in range(start + 1, min(start + 200, len(data))):
            if data[i] == 0x0A:
                return i

        # No newline found within reasonable distance
        return start + 50  # Estimate


class ObjectType(IntEnum):
    """APL2 ANS object types"""
    UNKNOWN = 0
    FUNCTION = 1      # Defined function
    VARIABLE = 2      # Variable
    OPERATOR = 3      # Defined operator
    GROUP = 4         # Group of objects
    NAMESPACE = 5     # Namespace
    CLASS = 6         # Class definition
    INSTANCE = 7      # Class instance


@dataclass
class ANSHeader:
    """ANS file header information"""
    magic: str
    version: str
    workspace_name: str
    object_count: int
    raw_bytes: bytes = field(default_factory=bytes, repr=False)


@dataclass
class ANSObject:
    """Represents an object in the ANS file"""
    name: str
    obj_type: ObjectType
    size: int
    offset: int
    flags: int = 0
    raw_data: bytes = field(default_factory=bytes, repr=False)
    decoded_content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APLFunction:
    """Decoded APL function"""
    name: str
    header: str                    # Function header line
    lines: List[str]               # Source code lines
    result_var: Optional[str] = None
    left_arg: Optional[str] = None
    right_arg: Optional[str] = None
    locals: List[str] = field(default_factory=list)
    is_operator: bool = False


@dataclass
class APLVariable:
    """Decoded APL variable"""
    name: str
    var_type: str                  # 'char', 'numeric', 'nested', 'mixed'
    rank: int                      # Number of dimensions
    shape: List[int]               # Dimensions
    data: Any                      # Actual data


def sanitize_workspace_name(name: str) -> str:
    """Sanitize workspace name for use as directory name."""
    if not name:
        return "WORKSPACE"
    # Remove null bytes and invalid path characters
    name = name.replace('\x00', '').replace(' ', '_').replace('/', '_')
    name = name.replace('\\', '_').replace(':', '_').replace('*', '_')
    name = name.replace('?', '_').replace('"', '_').replace('<', '_')
    name = name.replace('>', '_').replace('|', '_')
    name = ''.join(c for c in name if c.isprintable())
    return name if name else "WORKSPACE"


class ANSDecompiler:
    """
    Decompiler for IBM APL2 ANS (APL Namespace Save) files.

    Parses the binary format and extracts functions, variables,
    and other workspace objects into readable formats.
    """

    def __init__(self, filepath: str, verbose: bool = False):
        self.filepath = Path(filepath)
        self.verbose = verbose
        self.data: bytes = b''
        self.header: Optional[ANSHeader] = None
        self.objects: List[ANSObject] = []
        self.functions: List[APLFunction] = []
        self.variables: List[APLVariable] = []
        self.encoding = 'ascii'  # or 'ebcdic'
        self.bytecode_decoder = BytecodeDecoder(verbose=verbose)

    def log(self, msg: str):
        """Print verbose log message"""
        if self.verbose:
            print(f"[DEBUG] {msg}")

    def translate_apl(self, data: bytes) -> str:
        """
        Convert IBM APL2 charset bytes to Unicode APL symbols.

        Args:
            data: Raw bytes from the ANS file

        Returns:
            Unicode string with APL symbols
        """
        result = []
        translation_table = (APL2_EBCDIC_TO_UNICODE
                           if self.encoding == 'ebcdic'
                           else APL2_TO_UNICODE)

        for byte in data:
            if byte in translation_table:
                result.append(translation_table[byte])
            elif 0x20 <= byte <= 0x7E:
                # Printable ASCII
                result.append(chr(byte))
            else:
                # Unknown - use replacement or hex
                result.append(f'\\x{byte:02X}')

        return ''.join(result)

    def translate_apl_smart(self, data: bytes) -> str:
        """
        Intelligently translate bytes to APL characters.
        Tries to detect the encoding and handle mixed content.
        """
        # Try ASCII first for readable strings
        try:
            text = data.decode('ascii')
            # If it's all printable ASCII, return as-is
            if all(0x20 <= ord(c) <= 0x7E or c in '\n\r\t' for c in text):
                return text
        except UnicodeDecodeError:
            pass

        # Fall back to APL translation
        return self.translate_apl(data)

    def parse_header(self) -> ANSHeader:
        """
        Extract version and workspace name from file header.

        Returns:
            ANSHeader with parsed information
        """
        if len(self.data) < 0x50:
            raise ValueError("File too small to contain valid header")

        # Magic number at offset 0 - should be "ANS " or similar
        magic_raw = self.data[0:8]
        magic = magic_raw.rstrip(b'\x00').decode('ascii', errors='replace')

        # Version typically follows magic
        version = ""
        if b'1.00' in magic_raw or b'1.00' in self.data[0:16]:
            version = "1.00"

        # Workspace name at offset 0x30-0x39
        ws_name_raw = self.data[0x30:0x40]
        ws_name = ws_name_raw.rstrip(b'\x00').decode('ascii', errors='replace').strip()

        # Object count at offset 0x2C-0x2F (little-endian)
        obj_count = struct.unpack('<I', self.data[0x2C:0x30])[0]

        # Sanity check the object count
        if obj_count > 10000:
            # Try big-endian
            obj_count_be = struct.unpack('>I', self.data[0x2C:0x30])[0]
            if obj_count_be < obj_count:
                obj_count = obj_count_be

        self.log(f"Magic: {repr(magic)}")
        self.log(f"Version: {version}")
        self.log(f"Workspace name: {ws_name}")
        self.log(f"Object count: {obj_count}")

        self.header = ANSHeader(
            magic=magic,
            version=version,
            workspace_name=ws_name,
            object_count=obj_count,
            raw_bytes=self.data[0:0x50]
        )

        return self.header

    def find_object_table(self) -> int:
        """
        Locate the object table in the file.

        Returns:
            Offset to start of object table
        """
        # The object table often starts around 0x45F0 based on analysis
        # But we should search for it more dynamically

        # Look for patterns that indicate object entries
        # Object names are often uppercase ASCII followed by null padding

        potential_offsets = [0x45F0, 0x4000, 0x3000, 0x2000, 0x1000, 0x100]

        for offset in potential_offsets:
            if offset < len(self.data):
                # Check if this looks like an object table
                chunk = self.data[offset:offset + 100]
                # Look for readable names (uppercase letters)
                readable = sum(1 for b in chunk if 0x41 <= b <= 0x5A or 0x61 <= b <= 0x7A)
                if readable > 10:
                    self.log(f"Potential object table at 0x{offset:X}")
                    return offset

        # Default to searching the whole file
        return 0x100

    def parse_object_table(self) -> List[ANSObject]:
        """
        Build list of objects with their offsets and types.

        Returns:
            List of ANSObject entries
        """
        self.objects = []

        # Strategy: scan the file for function definitions and variable data
        # Look for the ∇ (nabla/del) symbol which marks functions

        # First, try to find object directory structure
        table_offset = self.find_object_table()

        # Scan for recognizable patterns
        self._scan_for_functions()
        self._scan_for_variables()

        self.log(f"Found {len(self.objects)} objects")
        return self.objects

    def _scan_for_functions(self):
        """Scan file for function definitions"""
        # Look for function markers in the binary
        # Functions typically have headers with ← and argument names

        i = 0
        while i < len(self.data) - 20:
            # Look for potential function start patterns
            # Check for readable name followed by specific patterns

            # Look for ∇ marker (various encodings)
            if self.data[i:i+1] in [b'\x99', b'\xC7', b'\x75']:  # Del in various encodings
                self._try_parse_function_at(i)

            # Also look for readable function-like structures
            # Function headers often have format: Z←NAME ARGS or similar
            chunk = self.data[i:i+50]
            if b'\xC0' in chunk or b'\xB1' in chunk:  # Left arrow in APL encodings
                self._try_parse_function_at(i)

            i += 1

    def _try_parse_function_at(self, offset: int) -> bool:
        """Try to parse a function starting at the given offset"""
        try:
            # Read a chunk and try to decode
            chunk = self.data[offset:offset + 2000]
            decoded = self.translate_apl_smart(chunk)

            # Look for function header pattern
            if '←' in decoded[:100] or '∇' in decoded[:10]:
                # This might be a function
                lines = decoded.split('\n')
                if lines:
                    # Extract function name from header
                    header = lines[0].strip()
                    name = self._extract_function_name(header)
                    if name and len(name) > 0 and name[0].isalpha():
                        obj = ANSObject(
                            name=name,
                            obj_type=ObjectType.FUNCTION,
                            size=len(chunk),
                            offset=offset,
                            raw_data=chunk
                        )
                        # Avoid duplicates
                        if not any(o.name == name and o.obj_type == ObjectType.FUNCTION
                                  for o in self.objects):
                            self.objects.append(obj)
                            self.log(f"Found function '{name}' at 0x{offset:X}")
                            return True
        except Exception as e:
            self.log(f"Error parsing at 0x{offset:X}: {e}")

        return False

    def _extract_function_name(self, header: str) -> str:
        """Extract function name from header line"""
        # Remove del symbol
        header = header.replace('∇', '').strip()

        # Common patterns:
        # Z←NAME ARGS
        # Z←ARGS NAME ARGS  (dyadic)
        # NAME ARGS

        if '←' in header:
            # Result ← expression
            parts = header.split('←')
            if len(parts) >= 2:
                expr = parts[1].strip()
                # First word after ← is usually the name or left arg
                words = expr.split()
                if words:
                    # Could be name or left arg - take first alphanumeric
                    for word in words:
                        word = word.strip(';').strip()
                        if word and word[0].isalpha():
                            return word
        else:
            # No result - name is first word
            words = header.split()
            for word in words:
                word = word.strip(';').strip()
                if word and word[0].isalpha():
                    return word

        return ""

    def _scan_for_variables(self):
        """Scan file for variable definitions"""
        # Variables are harder to detect - they're raw data
        # Look for array headers (shape information)
        pass

    def extract_object(self, obj: ANSObject) -> bytes:
        """
        Get raw bytes for an object.

        Args:
            obj: The object to extract

        Returns:
            Raw bytes of the object data
        """
        if obj.raw_data:
            return obj.raw_data

        if obj.offset >= 0 and obj.size > 0:
            return self.data[obj.offset:obj.offset + obj.size]

        return b''

    def decode_function(self, obj: ANSObject) -> APLFunction:
        """
        Parse function header and source lines.

        Uses the bytecode decoder to convert tokenized bytecode to APL source,
        then parses the function structure.

        Args:
            obj: ANSObject containing function data

        Returns:
            APLFunction with parsed structure
        """
        raw = self.extract_object(obj)

        # Try bytecode decoding first, fall back to character translation
        try:
            header, body_lines = self.bytecode_decoder.decode_function(raw)
            if header and len(header) > 2:
                decoded = header + '\n' + '\n'.join(body_lines)
            else:
                decoded = self.translate_apl_smart(raw)
        except Exception:
            decoded = self.translate_apl_smart(raw)

        lines = [l.rstrip() for l in decoded.split('\n') if l.strip()]

        # Parse header (first non-empty line)
        header = lines[0] if lines else ""

        # Extract components from header
        result_var = None
        left_arg = None
        right_arg = None
        locals_list = []

        # Parse: Z←L FUNC R;local1;local2
        header_clean = header.replace('∇', '').strip()

        # Split locals
        if ';' in header_clean:
            main_part, *local_parts = header_clean.split(';')
            locals_list = [l.strip() for l in local_parts if l.strip()]
            header_clean = main_part.strip()

        # Parse result and function
        if '←' in header_clean:
            result_part, func_part = header_clean.split('←', 1)
            result_var = result_part.strip()
            header_clean = func_part.strip()

        # Parse name and arguments
        words = header_clean.split()
        if len(words) >= 3:
            # Dyadic: LEFTARG NAME RIGHTARG
            left_arg = words[0]
            func_name = words[1]
            right_arg = words[2] if len(words) > 2 else None
        elif len(words) == 2:
            # Monadic: NAME RIGHTARG
            func_name = words[0]
            right_arg = words[1]
        elif len(words) == 1:
            # Niladic
            func_name = words[0]
        else:
            func_name = obj.name

        func = APLFunction(
            name=func_name,
            header=header,
            lines=lines[1:] if len(lines) > 1 else [],  # Exclude header
            result_var=result_var,
            left_arg=left_arg,
            right_arg=right_arg,
            locals=locals_list
        )

        return func

    def decode_variable(self, obj: ANSObject) -> APLVariable:
        """
        Parse variable type, shape, and data.

        Args:
            obj: ANSObject containing variable data

        Returns:
            APLVariable with parsed structure
        """
        raw = self.extract_object(obj)

        # Variable format typically includes:
        # - Type indicator
        # - Rank (number of dimensions)
        # - Shape (dimensions)
        # - Data

        # Try to parse the structure
        if len(raw) < 4:
            return APLVariable(
                name=obj.name,
                var_type='unknown',
                rank=0,
                shape=[],
                data=raw
            )

        # First bytes often indicate type
        type_byte = raw[0]

        if type_byte in [0x01, 0x02]:  # Numeric
            var_type = 'numeric'
        elif type_byte in [0x03, 0x04]:  # Character
            var_type = 'char'
        elif type_byte in [0x05, 0x06]:  # Nested
            var_type = 'nested'
        else:
            var_type = 'unknown'

        # Try to extract shape
        rank = raw[1] if len(raw) > 1 else 0
        shape = []

        # Decode data based on type
        if var_type == 'char':
            data = self.translate_apl_smart(raw[4:])
        else:
            data = raw[4:]

        return APLVariable(
            name=obj.name,
            var_type=var_type,
            rank=rank,
            shape=shape,
            data=data
        )

    def decompile(self) -> Tuple[List[APLFunction], List[APLVariable]]:
        """
        Main entry point - parse the ANS file and extract all objects.

        Uses improved ANSBinaryParser and FunctionBoundaryDetector for better results.

        Returns:
            Tuple of (functions, variables)
        """
        # Read file
        self.log(f"Reading {self.filepath}")
        with open(self.filepath, 'rb') as f:
            self.data = f.read()

        self.log(f"File size: {len(self.data)} bytes")

        # Detect encoding
        self._detect_encoding()

        # Try new binary parser first
        try:
            binary_parser = ANSBinaryParser(verbose=self.verbose)
            structure = binary_parser.parse(self.data)

            # Update header with parsed information
            self.header = ANSHeader(
                magic=structure['magic'],
                version=structure['version'],
                workspace_name=structure['workspace_name'],
                object_count=len(structure['objects']),
                raw_bytes=self.data[0:0x100]
            )

            # Use FunctionBoundaryDetector for each object
            boundary_detector = FunctionBoundaryDetector(verbose=self.verbose)

            for obj_data in structure['objects']:
                if obj_data['type'] == 1 or obj_data['type'] == 0:  # Function or unknown (might be function)
                    # Try to find function boundaries
                    raw_data = obj_data['data']
                    boundaries_list = boundary_detector.find_functions(raw_data)

                    if boundaries_list and len(boundaries_list) > 0:
                        # Use first boundary set
                        boundaries = boundaries_list[0]

                        # Create ANSObject
                        ans_obj = ANSObject(
                            name=obj_data['name'],
                            obj_type=ObjectType.FUNCTION,
                            size=obj_data['size'],
                            offset=obj_data['offset'],
                            raw_data=raw_data
                        )

                        # Decode with boundaries
                        try:
                            func = self._decode_function_with_boundaries(ans_obj, boundaries)
                            if func and func.name:
                                self.functions.append(func)
                        except Exception as e:
                            self.log(f"Error decoding function {obj_data['name']}: {e}")
                            # Fallback to old method
                            try:
                                func = self.decode_function(ans_obj)
                                if func and func.name:
                                    self.functions.append(func)
                            except:
                                pass

                elif obj_data['type'] == 2:  # Variable
                    ans_obj = ANSObject(
                        name=obj_data['name'],
                        obj_type=ObjectType.VARIABLE,
                        size=obj_data['size'],
                        offset=obj_data['offset'],
                        raw_data=obj_data['data']
                    )
                    try:
                        var = self.decode_variable(ans_obj)
                        if var:
                            self.variables.append(var)
                    except Exception as e:
                        self.log(f"Error decoding variable {obj_data['name']}: {e}")

        except Exception as e:
            self.log(f"Binary parser failed: {e}, falling back to old method")
            # Fallback to original method
            self.parse_header()
            self.parse_object_table()

            for obj in self.objects:
                if obj.obj_type == ObjectType.FUNCTION:
                    try:
                        func = self.decode_function(obj)
                        if func and func.name:
                            self.functions.append(func)
                    except Exception as e:
                        self.log(f"Error decoding function: {e}")
                elif obj.obj_type == ObjectType.VARIABLE:
                    try:
                        var = self.decode_variable(obj)
                        if var:
                            self.variables.append(var)
                    except Exception as e:
                        self.log(f"Error decoding variable: {e}")

        self.log(f"Decoded {len(self.functions)} functions, {len(self.variables)} variables")

        return self.functions, self.variables

    def _decode_function_with_boundaries(self, obj: ANSObject, boundaries: Dict[str, int]) -> APLFunction:
        """
        Decode function using detected boundaries.

        Args:
            obj: ANSObject containing function data
            boundaries: Dict with 'start', 'end', 'header_end' keys

        Returns:
            APLFunction with parsed structure
        """
        raw = obj.raw_data

        # Use enhanced decoder with boundaries
        header, body_lines = self.bytecode_decoder.decode_with_boundaries(raw, boundaries)

        # Extract components from header
        result_var = None
        left_arg = None
        right_arg = None
        locals_list = []

        # Parse header: Z←L FUNC R;local1;local2
        header_clean = header.replace('∇', '').strip()

        # Split locals
        if ';' in header_clean:
            main_part, *local_parts = header_clean.split(';')
            locals_list = [l.strip() for l in local_parts if l.strip()]
            header_clean = main_part.strip()

        # Parse result and function
        if '←' in header_clean:
            result_part, func_part = header_clean.split('←', 1)
            result_var = result_part.strip()
            header_clean = func_part.strip()

        # Parse name and arguments
        words = header_clean.split()
        if len(words) >= 3:
            # Dyadic: LEFTARG NAME RIGHTARG
            left_arg = words[0]
            func_name = words[1]
            right_arg = words[2] if len(words) > 2 else None
        elif len(words) == 2:
            # Monadic: NAME RIGHTARG
            func_name = words[0]
            right_arg = words[1]
        elif len(words) == 1:
            # Niladic
            func_name = words[0]
        else:
            func_name = obj.name if obj.name else 'unknown'

        func = APLFunction(
            name=func_name,
            header=header,
            lines=body_lines,
            result_var=result_var,
            left_arg=left_arg,
            right_arg=right_arg,
            locals=locals_list
        )

        return func

    def _detect_encoding(self):
        """Detect whether file uses ASCII or EBCDIC encoding"""
        # Check for EBCDIC patterns
        sample = self.data[0:1000]

        # EBCDIC has letters in 0x81-0xA9 and 0xC1-0xE9 ranges
        ebcdic_letters = sum(1 for b in sample
                           if 0x81 <= b <= 0x89 or 0x91 <= b <= 0x99
                           or 0xA2 <= b <= 0xA9 or 0xC1 <= b <= 0xC9
                           or 0xD1 <= b <= 0xD9 or 0xE2 <= b <= 0xE9)

        # ASCII has letters in 0x41-0x5A and 0x61-0x7A
        ascii_letters = sum(1 for b in sample
                          if 0x41 <= b <= 0x5A or 0x61 <= b <= 0x7A)

        if ebcdic_letters > ascii_letters * 2:
            self.encoding = 'ebcdic'
            self.log("Detected EBCDIC encoding")
        else:
            self.encoding = 'ascii'
            self.log("Detected ASCII encoding")

    def export_text(self, output_dir: Path, combined: bool = True,
                   individual: bool = True) -> List[Path]:
        """
        Output as readable text files.

        Args:
            output_dir: Directory to write output files
            combined: Whether to create a single combined file
            individual: Whether to create individual function files

        Returns:
            List of created file paths
        """
        created_files = []

        ws_name = sanitize_workspace_name(
            self.header.workspace_name if self.header else "WORKSPACE"
        )

        # Create directory structure
        ws_dir = output_dir / ws_name
        ws_dir.mkdir(parents=True, exist_ok=True)

        if individual:
            func_dir = ws_dir / "functions"
            func_dir.mkdir(exist_ok=True)

            for func in self.functions:
                # Sanitize function name for filename
                safe_name = sanitize_workspace_name(func.name)
                if not safe_name or len(safe_name) > 100:
                    safe_name = f"func_{len(created_files)}"
                func_file = func_dir / f"{safe_name}.apl"

                content = self._format_function(func)
                func_file.write_text(content, encoding='utf-8')
                created_files.append(func_file)

        if combined:
            combined_file = ws_dir / "workspace.apl"

            lines = [
                f"⍝ === Workspace: {ws_name} ===",
                f"⍝ Decompiled from: {self.filepath.name}",
                f"⍝ Functions: {len(self.functions)}",
                f"⍝ Variables: {len(self.variables)}",
                "",
            ]

            for func in self.functions:
                lines.append("")
                lines.append(self._format_function(func))

            combined_file.write_text('\n'.join(lines), encoding='utf-8')
            created_files.append(combined_file)

        return created_files

    def _format_function(self, func: APLFunction) -> str:
        """Format a function for text output"""
        lines = [
            f"⍝ === Function: {func.name} ===",
            f"∇ {func.header.replace('∇', '').strip()}"
        ]

        for line in func.lines:
            # Skip closing del if present
            if line.strip() == '∇':
                continue
            lines.append(f"  {line}")

        lines.append("∇")
        lines.append("")

        return '\n'.join(lines)

    def export_json(self, output_dir: Path) -> Path:
        """
        Output as structured JSON with full metadata.

        Args:
            output_dir: Directory to write output file

        Returns:
            Path to created JSON file
        """
        ws_name = sanitize_workspace_name(
            self.header.workspace_name if self.header else "WORKSPACE"
        )

        ws_dir = output_dir / ws_name
        ws_dir.mkdir(parents=True, exist_ok=True)

        json_file = ws_dir / "workspace.json"

        data = {
            "workspace": {
                "name": ws_name,
                "source_file": str(self.filepath),
                "magic": self.header.magic if self.header else "",
                "version": self.header.version if self.header else "",
                "object_count": self.header.object_count if self.header else 0,
            },
            "functions": [],
            "variables": [],
            "statistics": {
                "total_functions": len(self.functions),
                "total_variables": len(self.variables),
                "file_size": len(self.data),
            }
        }

        for func in self.functions:
            data["functions"].append({
                "name": func.name,
                "header": func.header,
                "lines": func.lines,
                "result": func.result_var,
                "left_arg": func.left_arg,
                "right_arg": func.right_arg,
                "locals": func.locals,
                "is_operator": func.is_operator,
                "line_count": len(func.lines),
            })

        for var in self.variables:
            var_data = {
                "name": var.name,
                "type": var.var_type,
                "rank": var.rank,
                "shape": var.shape,
            }
            # Only include data if it's serializable
            if isinstance(var.data, (str, int, float, list)):
                var_data["data"] = var.data
            elif isinstance(var.data, bytes):
                var_data["data_hex"] = var.data.hex()
            data["variables"].append(var_data)

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return json_file


def decompile_file(filepath: str, output_dir: str = "./output",
                   verbose: bool = False) -> Dict[str, Any]:
    """
    Convenience function to decompile an ANS file.

    Args:
        filepath: Path to the ANS file
        output_dir: Output directory
        verbose: Whether to print debug info

    Returns:
        Dictionary with results
    """
    decompiler = ANSDecompiler(filepath, verbose=verbose)
    functions, variables = decompiler.decompile()

    output_path = Path(output_dir)

    # Export all formats
    text_files = decompiler.export_text(output_path)
    json_file = decompiler.export_json(output_path)

    return {
        "functions": len(functions),
        "variables": len(variables),
        "output_files": [str(f) for f in text_files] + [str(json_file)],
        "workspace_name": decompiler.header.workspace_name if decompiler.header else "UNKNOWN",
    }


def main():
    """Main entry point for CLI usage"""
    parser = argparse.ArgumentParser(
        description="Decompile IBM APL2 ANS workspace files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s SQL.ans                    # Decompile to ./output/SQL/
  %(prog)s SQL.ans -o mydir           # Custom output directory
  %(prog)s *.ans                      # Batch process multiple files
  %(prog)s SQL.ans -v                 # Verbose output for debugging
        """
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="ANS file(s) to decompile"
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
        "--no-individual",
        action="store_true",
        help="Don't create individual function files"
    )

    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Don't create combined workspace file"
    )

    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Don't create JSON output"
    )

    parser.add_argument(
        "--transpile",
        action="store_true",
        help="Also generate Python code from the decompiled APL"
    )

    parser.add_argument(
        "--python-only",
        action="store_true",
        help="Only generate Python output (implies --transpile, skips APL files)"
    )

    args = parser.parse_args()

    # --python-only implies --transpile
    if args.python_only:
        args.transpile = True

    # Process each file
    results = []
    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}", file=sys.stderr)
            continue

        print(f"Decompiling: {filepath}")

        try:
            decompiler = ANSDecompiler(filepath, verbose=args.verbose)
            functions, variables = decompiler.decompile()

            output_path = Path(args.output)
            created_files = []

            # Export based on options
            if not args.no_individual or not args.no_combined:
                text_files = decompiler.export_text(
                    output_path,
                    combined=not args.no_combined,
                    individual=not args.no_individual
                )
                created_files.extend(text_files)

            if not args.no_json:
                json_file = decompiler.export_json(output_path)
                created_files.append(json_file)

            ws_name = decompiler.header.workspace_name if decompiler.header else "UNKNOWN"
            ws_name_safe = sanitize_workspace_name(ws_name)

            # Transpile to Python if requested
            python_file = None
            if args.transpile:
                try:
                    from apl_transpiler import APLTranspiler, PythonGenerator, APLLexer, APLParser

                    # Build APL source from decompiled functions
                    apl_lines = []
                    for func in functions:
                        apl_lines.append(f"∇ {func.header.replace('∇', '').strip()}")
                        for line in func.lines:
                            if line.strip() != '∇':
                                apl_lines.append(f"  {line}")
                        apl_lines.append("∇")
                        apl_lines.append("")

                    apl_source = '\n'.join(apl_lines)

                    # Transpile
                    lexer = APLLexer(apl_source)
                    tokens = lexer.tokenize()
                    parser = APLParser(tokens)
                    program = parser.parse_program()
                    generator = PythonGenerator()
                    python_code = generator.generate_program(program)

                    # Write Python output
                    python_dir = output_path / ws_name_safe / "python"
                    python_dir.mkdir(parents=True, exist_ok=True)

                    python_file = python_dir / "workspace.py"
                    python_file.write_text(python_code, encoding='utf-8')
                    created_files.append(python_file)

                    # Write __init__.py
                    init_file = python_dir / "__init__.py"
                    init_file.write_text(
                        f'"""Transpiled APL workspace: {ws_name}"""\nfrom .workspace import *\n',
                        encoding='utf-8'
                    )
                    created_files.append(init_file)

                    # Copy primitives library if it exists
                    primitives_src = Path(__file__).parent / "apl_primitives.py"
                    if primitives_src.exists():
                        primitives_dst = python_dir / "apl_primitives.py"
                        primitives_dst.write_text(primitives_src.read_text(), encoding='utf-8')
                        created_files.append(primitives_dst)

                    print(f"  Python: {python_file}")

                except ImportError as e:
                    print(f"  Warning: Could not transpile - {e}", file=sys.stderr)
                except Exception as e:
                    print(f"  Warning: Transpilation failed - {e}", file=sys.stderr)
                    if args.verbose:
                        import traceback
                        traceback.print_exc()

            results.append({
                "file": filepath,
                "workspace": ws_name,
                "functions": len(functions),
                "variables": len(variables),
                "output_files": created_files,
                "python_file": str(python_file) if python_file else None,
                "success": True
            })

            print(f"  Workspace: {ws_name}")
            print(f"  Functions: {len(functions)}")
            print(f"  Variables: {len(variables)}")
            print(f"  Output: {output_path / ws_name_safe}")

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


if __name__ == "__main__":
    sys.exit(main())
