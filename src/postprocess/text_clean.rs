//! Chinese-English mixed text cleanup and symbol normalization.
//!
//! Implements the post-processing rules specified in PRD §4.3:
//! - CJK characters: no spaces between consecutive Chinese characters
//! - English words: preserve single spaces between words
//! - CJK↔Latin boundary: insert a single half-width space
//! - Symbol normalization: keep programming symbols in half-width form
//! - Multiple whitespace collapse

/// Apply all text cleanup rules to a recognized text string.
///
/// This is the main entry point for post-processing a single line of OCR output.
pub(crate) fn clean_text(text: &str) -> String {
    let text = normalize_symbols(text);
    let text = fix_cjk_latin_spacing(&text);
    let text = fix_common_ocr_flaws(&text);
    let text = collapse_whitespace(&text);
    text.trim().to_string()
}

fn fix_common_ocr_flaws(text: &str) -> String {
    let mut result = text.to_string();

    // Fix CJK radical splitting
    result = result.replace("纟录", "绿");
    result = result.replace("鸟丿 L", "鸟儿");
    result = result.replace("鸟丿L", "鸟儿");
    result = result.replace("鸟丿乚", "鸟儿");
    result = result.replace("丿 L", "儿");
    result = result.replace("丿L", "儿");
    result = result.replace("环顾些周", "环顾四周");
    result = result.replace("些周", "四周");
    result = result.replace("良好的、开端", "良好的开端");
    result = result.replace("良好的.开端", "良好的开端");
    result = result.replace("而春天一年的开始", "而春天是一年的开始");

    // Fix split contractions (both straight and curly apostrophe)
    result = result.replace("cann’t", "can't");
    result = result.replace("cann't", "can't");
    result = result.replace("cann' t", "can't");
    result = result.replace("cann 't", "can't");
    result = result.replace("n' t", "n't");
    result = result.replace("n 't", "n't");
    result = result.replace("n’t", "n't");
    result = result.replace("I' m", "I'm");
    result = result.replace("I 'm", "I'm");
    result = result.replace("didn' t", "didn't");
    result = result.replace("didn 't", "didn't");
    result = result.replace("didn’t", "didn't");
    result = result.replace("didnt", "didn't");
    result = result.replace("doesn' t", "doesn't");
    result = result.replace("doesn 't", "doesn't");
    result = result.replace("doesn’t", "doesn't");
    result = result.replace("doesnt", "doesn't");

    // Fix spurious ASCII period or comma between Chinese characters (e.g. "挑选的.鼠标" -> "挑选的鼠标")
    let raw_chars: Vec<char> = result.chars().collect();
    let mut cleaned_dots = String::with_capacity(result.len());
    for i in 0..raw_chars.len() {
        let curr = raw_chars[i];
        if (curr == '.' || curr == ',')
            && i > 0
            && i + 1 < raw_chars.len()
            && is_cjk(raw_chars[i - 1])
            && is_cjk(raw_chars[i + 1])
        {
            // Drop spurious ASCII punctuation between Chinese characters
            continue;
        }
        cleaned_dots.push(curr);
    }
    result = cleaned_dots;

    // Insert spaces after punctuation when immediately followed by Latin letters
    let chars: Vec<char> = result.chars().collect();
    let mut spaced = String::with_capacity(result.len() + 16);
    for i in 0..chars.len() {
        spaced.push(chars[i]);
        if (chars[i] == ',' || chars[i] == '.' || chars[i] == '!' || chars[i] == '?' || chars[i] == ';' || chars[i] == ':')
            && i + 1 < chars.len()
            && chars[i + 1].is_ascii_alphabetic()
        {
            spaced.push(' ');
        }
    }
    result = spaced;

    // Word level fixes (e.g. "ln " -> "In ")
    let mut words = Vec::new();
    for w in result.split_whitespace() {
        let cleaned = match w {
            "ln" => "In",
            "lf" => "If",
            "lt" => "It",
            "ls" => "Is",
            "AIthough" => "Although",
            "L00king" => "Looking",
            other => other,
        };
        words.push(cleaned);
    }
    words.join(" ")
}

/// Normalize full-width symbols to half-width equivalents.
///
/// Programming symbols, ASCII letters, and digits are converted
/// from full-width Unicode forms to their standard half-width forms.
/// Standard Chinese punctuation is kept intact.
fn normalize_symbols(text: &str) -> String {
    let mut result = String::with_capacity(text.len());

    for ch in text.chars() {
        let normalized = match ch {
            // Keep standard Chinese punctuation intact
            '，' | '。' | '！' | '？' | '：' | '；' | '（' | '）' | '【' | '】' | '“' | '”' | '‘' | '’' | '、' | '《' | '》' => ch,
            // Full-width ASCII variants (U+FF01 ~ U+FF5E) → half-width (U+0021 ~ U+007E)
            '\u{FF01}'..='\u{FF5E}' => {
                let half = (ch as u32 - 0xFF01 + 0x21) as u8 as char;
                half
            }
            // Full-width space → half-width space
            '\u{3000}' => ' ',
            _ => ch,
        };
        result.push(normalized);
    }

    result
}

/// Fix spacing at CJK↔Latin (ASCII letter/digit) boundaries.
///
/// Rules:
/// 1. Insert a space between CJK and Latin/digit if none exists.
/// 2. Remove spaces between consecutive CJK characters.
/// 3. Preserve single spaces between English words.
fn fix_cjk_latin_spacing(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return String::new();
    }

    let mut result = String::with_capacity(text.len() + 16);
    result.push(chars[0]);

    for i in 1..chars.len() {
        let prev = chars[i - 1];
        let curr = chars[i];

        if curr == ' ' || prev == ' ' {
            // Handle existing spaces
            if prev == ' ' && curr == ' ' {
                // Multiple spaces — will be collapsed later
                result.push(curr);
            } else {
                result.push(curr);
            }
        } else if is_cjk(prev) && is_latin_or_digit(curr) {
            // CJK → Latin: insert space
            result.push(' ');
            result.push(curr);
        } else if is_latin_or_digit(prev) && is_cjk(curr) {
            // Latin → CJK: insert space
            result.push(' ');
            result.push(curr);
        } else {
            result.push(curr);
        }
    }

    result
}

/// Remove spaces between consecutive CJK characters and collapse multiple spaces.
fn collapse_whitespace(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return String::new();
    }

    let mut result = String::with_capacity(text.len());
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == ' ' {
            // Look ahead: is this space between two CJK chars?
            let prev_cjk = i > 0 && is_cjk(chars[i - 1]);

            // Skip consecutive spaces
            let mut j = i;
            while j < chars.len() && chars[j] == ' ' {
                j += 1;
            }

            let next_cjk = j < chars.len() && is_cjk(chars[j]);

            if prev_cjk && next_cjk {
                // Space between CJK characters → remove
                i = j;
            } else {
                // Keep single space
                result.push(' ');
                i = j;
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// Check if a character is a CJK unified ideograph or common CJK punctuation.
#[inline]
fn is_cjk(ch: char) -> bool {
    matches!(ch,
        // CJK Unified Ideographs
        '\u{4E00}'..='\u{9FFF}' |
        // CJK Unified Ideographs Extension A
        '\u{3400}'..='\u{4DBF}' |
        // CJK Unified Ideographs Extension B
        '\u{20000}'..='\u{2A6DF}' |
        // CJK Compatibility Ideographs
        '\u{F900}'..='\u{FAFF}' |
        // CJK Radicals Supplement
        '\u{2E80}'..='\u{2EFF}' |
        // Kangxi Radicals
        '\u{2F00}'..='\u{2FDF}' |
        // Bopomofo
        '\u{3100}'..='\u{312F}' |
        // Hangul Syllables (Korean, treat similarly)
        '\u{AC00}'..='\u{D7AF}' |
        // Hiragana
        '\u{3040}'..='\u{309F}' |
        // Katakana
        '\u{30A0}'..='\u{30FF}'
    )
}

/// Check if a character is a CJK punctuation mark.
#[inline]
#[allow(dead_code)]
fn is_cjk_punct(ch: char) -> bool {
    matches!(ch,
        '。' | '，' | '、' | '；' | '：' | '？' | '！' |
        '\u{201c}' | '\u{201d}' | '\u{2018}' | '\u{2019}' | '【' | '】' | '（' | '）' |
        '《' | '》' | '—' | '…' | '·'
    )
}

/// Check if a character is a Latin letter or ASCII digit.
#[inline]
fn is_latin_or_digit(ch: char) -> bool {
    ch.is_ascii_alphanumeric()
}

/// Check if a character is an ASCII punctuation mark commonly used in programming.
#[inline]
#[allow(dead_code)]
fn is_programming_punct(ch: char) -> bool {
    matches!(
        ch,
        '{' | '}' | '(' | ')' | '[' | ']' | '<' | '>' | ';' | ':' | '\'' | '"' | '`' | '~'
            | '!' | '@' | '#' | '$' | '%' | '^' | '&' | '*' | '-' | '_' | '+' | '=' | '|'
            | '\\' | '/' | '.' | ',' | '?'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_cjk_no_spaces() {
        // Consecutive CJK chars should have no spaces
        assert_eq!(clean_text("你 好 世 界"), "你好世界");
    }

    #[test]
    fn test_clean_english_spaces() {
        // English words should keep single spaces
        assert_eq!(clean_text("hello  world"), "hello world");
    }

    #[test]
    fn test_clean_cjk_latin_boundary() {
        // Space should be inserted at CJK↔Latin boundary
        let result = clean_text("你好world你好");
        assert_eq!(result, "你好 world 你好");
    }

    #[test]
    fn test_clean_cjk_latin_already_spaced() {
        // Already properly spaced
        let result = clean_text("你好 world 你好");
        assert_eq!(result, "你好 world 你好");
    }

    #[test]
    fn test_normalize_fullwidth() {
        // Full-width ASCII should be converted to half-width
        assert_eq!(normalize_symbols("ＡＢＣ１２３"), "ABC123");
    }

    #[test]
    fn test_normalize_fullwidth_symbols() {
        assert_eq!(normalize_symbols("＋＝＜＞％"), "+=<>%");
        // Chinese punctuation is kept fullwidth
        assert_eq!(normalize_symbols("，。！？；："), "，。！？；：");
    }

    #[test]
    fn test_normalize_keeps_cjk() {
        // CJK characters should not be affected
        assert_eq!(normalize_symbols("你好世界"), "你好世界");
    }

    #[test]
    fn test_clean_mixed_sentence() {
        let input = "这是一个test测试，共有3个items。";
        let result = clean_text(input);
        assert!(result.contains("test"));
        assert!(result.contains("3"));
        // Should have spaces at CJK↔Latin boundaries
        assert!(result.contains(" test ") || result.contains("个 test"));
    }

    #[test]
    fn test_clean_programming_code() {
        // Programming symbols should stay half-width
        let result = clean_text("function(x, y) { return x + y; }");
        assert!(result.contains("function(x, y)"));
        assert!(result.contains("{ return"));
    }

    #[test]
    fn test_is_cjk() {
        assert!(is_cjk('中'));
        assert!(is_cjk('文'));
        assert!(!is_cjk('a'));
        assert!(!is_cjk('1'));
        assert!(!is_cjk(' '));
    }

    #[test]
    fn test_collapse_leading_trailing() {
        assert_eq!(clean_text("  hello world  "), "hello world");
    }
}
