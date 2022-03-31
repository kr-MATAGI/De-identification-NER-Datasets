from enum import Enum

class WIKI_RE(Enum):
    CITE = r"\{\{[^\}]+\}\}"

    FILE = r"(\[\[)파일:[^\]]+(\]\])"
    MEDIA = r"(\[\[)미디어:[^\]]+(\]\])"

    FONT_SHAPE_5 = r"(''''')[^']+(''''')"
    FONT_SHAPE_3 = r"(''')[^']+(''')"
    FONT_SHAPE_2 = r"('')[^']+('')"

    SPECIAL_CHAR = r"&[a-zA-Z]+;"
    SUBP_SCRIPT = r"<su(b|p)>[^(<su(b|p)>))]+</su(b|p)>"

    MATH_TAG = r"<math>[^<]+</math>"
    SPAN_TAG = r"<span (?!</span>).+</span>"
    SMALL_TAG = r"<small>[^<]*</small>"
    BIG_TAG = r"<big>[^<]*</big>"
    NO_INCLUDE_TAG = r"<noinclude>[^<]*</noinclude>"
    NO_WIKI_TAG = r"<nowiki>[^<]*</nowiki>"
    ONLY_INCLUDE_TAG = r"<onlyinclude>[^<]*</onlyinclude>"
    INCLUDE_ONLY_TAG = r"<noinclude>[^<]*</noinclude>"

    REDIRECT = r"#넘겨주기 (\[\[)[^\]]+(\]\])"
    COMMENT = r"(<!--).+(-->)"
    PRE = "(<pre>)|(</pre>)"

    REF = r"(<ref>).+(</ref>)"
    REF_2 = r"<ref .+>[^<]+(</ref>)"
    REF_3 = r"<ref[^=]+=[^/]+/>"
    REF_4 = r"ref.+/ref"
    REF_5 = r"<ref>.+"

    BR = r"<br />"

    FREE_LINK_BASIC = r"(\[\[)[^\]]+(\]\])"
    FREE_LINK_ALT = r"(\[\[)([^\]]+\|[^\]]+)(\]\])"
    FREE_LINK_LHS = r"(\[\[)[^\]]+\|"
    FREE_LINK_OPEN = r"(\[\[)"
    FREE_LINK_CLOSED = r"(\]\])"

    EXT_LINK_ALT = r"\[[^\]]+ [^\]]+\]"
    EXT_LINK_ALT_LHS = r"\[[^\]]+\s"
    EXT_LINK = r"\[[^\]]+\]"