/// cloak-nerd — Token-classification PII inference sidecar for Cloak
///
/// Reads NDJSON requests from stdin, returns NDJSON responses on stdout.
/// Each input line: {"text": "...", "labels": ["NAME", "ADDRESS", "USERNAME"]}
/// Each output line: {"entities": [{"start":5,"end":15,"type":"NAME","score":0.95},...]}
///
/// The labels in the request are treated as a filter/allow-list over the
/// classifier's fixed output classes (Core-3: NAME, ADDRESS, USERNAME).
use anyhow::{Context, Result};
use clap::Parser;
use ort::session::Session;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(
    name = "cloak-nerd",
    about = "Token-classification PII inference sidecar"
)]
struct Cli {
    /// Path to model.onnx
    #[arg(long)]
    model: PathBuf,

    /// Path to tokenizer.json
    #[arg(long)]
    tokenizer: PathBuf,

    /// Path to model_config.json (id2label, max_len, model_type, onnx_inputs)
    #[arg(long)]
    config: PathBuf,

    /// Entity score threshold (0.0–1.0), applied to per-token softmax max
    #[arg(long, default_value = "0.5")]
    threshold: f32,

    /// Maximum number of spans per text
    #[arg(long, default_value = "32")]
    max_spans: usize,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct ModelConfig {
    model_type: String,
    id2label: HashMap<String, String>,
    label2id: HashMap<String, usize>, // inverse map — part of config contract
    max_len: usize,
    onnx_inputs: Vec<String>,
    #[serde(default)]
    onnx_outputs: Vec<String>,
}

impl ModelConfig {
    /// Parse id2label where keys are string-encoded integers.
    fn id2label_map(&self) -> HashMap<usize, String> {
        self.id2label
            .iter()
            .filter_map(|(k, v)| k.parse::<usize>().ok().map(|i| (i, v.clone())))
            .collect()
    }
}

#[derive(Deserialize)]
struct Request {
    text: String,
    labels: Vec<String>,
}

#[derive(Serialize)]
struct Response {
    entities: Vec<Entity>,
}

#[derive(Serialize)]
struct Entity {
    start: usize,
    end: usize,
    #[serde(rename = "type")]
    entity_type: String,
    score: f32,
}

/// A span being built during BIO decoding.
struct OpenSpan {
    entity_type: String,
    start_byte: usize,
    end_byte: usize,
    min_score: f32,
}

/// Decode BIO-tagged subword tokens into entity spans.
///
/// The key insight: because training labels *every* subword of an entity
/// (B- on first, I- on rest), we can merge entity spans using byte offsets
/// alone — no word-boundary reconstruction is needed.
///
/// Convention (§6.3): contiguous non-O tokens form one entity; `B-X` starts
/// one, `I-X` continues it. Any change (O or different type) closes the span.
/// Score = min of per-token max-class probabilities (conservative).
fn decode_bio(
    token_label_ids: &[usize],
    token_byte_starts: &[usize],
    token_byte_ends: &[usize],
    token_scores: &[f32],
    id2label: &HashMap<usize, String>,
    threshold: f32,
    allow_types: &[String],
) -> Vec<(usize, usize, String, f32)> {
    let allow_set: std::collections::HashSet<&str> =
        allow_types.iter().map(|s| s.as_str()).collect();

    let mut entities: Vec<(usize, usize, String, f32)> = Vec::new();
    let mut open: Option<OpenSpan> = None;

    for i in 0..token_label_ids.len() {
        // Skip special tokens (byte offsets 0-0).
        if token_byte_starts[i] == token_byte_ends[i] {
            if let Some(ref span) = open {
                entities.push((
                    span.start_byte,
                    span.end_byte,
                    span.entity_type.clone(),
                    span.min_score,
                ));
                open = None;
            }
            continue;
        }

        let score = token_scores[i];
        let label_str = id2label.get(&token_label_ids[i]).map(String::as_str);

        // Below threshold → treat as O.
        let effective_label = if score >= threshold {
            label_str.unwrap_or("O")
        } else {
            "O"
        };

        if effective_label == "O" {
            if let Some(ref span) = open {
                entities.push((
                    span.start_byte,
                    span.end_byte,
                    span.entity_type.clone(),
                    span.min_score,
                ));
                open = None;
            }
            continue;
        }

        // Extract entity type from BIO tag.
        let etype = if effective_label.starts_with("B-") || effective_label.starts_with("I-") {
            &effective_label[2..]
        } else {
            // Not a BIO tag — close any open span and skip.
            if let Some(ref span) = open {
                entities.push((
                    span.start_byte,
                    span.end_byte,
                    span.entity_type.clone(),
                    span.min_score,
                ));
                open = None;
            }
            continue;
        };

        // Only emit types in the allow-list.
        if !allow_set.contains(etype) {
            if let Some(ref span) = open {
                entities.push((
                    span.start_byte,
                    span.end_byte,
                    span.entity_type.clone(),
                    span.min_score,
                ));
                open = None;
            }
            continue;
        }

        let byte_start = token_byte_starts[i];
        let byte_end = token_byte_ends[i];

        match open.as_mut() {
            Some(ref mut span) if span.entity_type == etype && effective_label.starts_with("I") => {
                // Continue existing span.
                span.end_byte = byte_end;
                span.min_score = span.min_score.min(score);
            }
            Some(ref span) => {
                // Different type or B- on an already-open span → close old, start new.
                entities.push((
                    span.start_byte,
                    span.end_byte,
                    span.entity_type.clone(),
                    span.min_score,
                ));
                open = Some(OpenSpan {
                    entity_type: etype.to_string(),
                    start_byte: byte_start,
                    end_byte: byte_end,
                    min_score: score,
                });
            }
            None => {
                open = Some(OpenSpan {
                    entity_type: etype.to_string(),
                    start_byte: byte_start,
                    end_byte: byte_end,
                    min_score: score,
                });
            }
        }
    }

    // Flush any remaining open span.
    if let Some(ref span) = open {
        entities.push((
            span.start_byte,
            span.end_byte,
            span.entity_type.clone(),
            span.min_score,
        ));
    }

    entities
}

fn run_inference(
    text: &str,
    labels: &[String],
    session: &mut Session,
    tokenizer: &Tokenizer,
    cfg: &ModelConfig,
    threshold: f32,
    max_spans: usize,
) -> Result<Response> {
    // 1. Tokenize with byte offsets. add_special_tokens=true: the model was
    // trained with [CLS]/[SEP]; omitting them shifts every prediction.
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;

    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let seq_len = input_ids.len();
    let attention_mask: Vec<i64> = vec![1; seq_len];

    // Truncate if too long (tokenizer's built-in truncation was already
    // applied; this is a safety clamp).
    let max_len = std::cmp::min(seq_len, cfg.max_len);
    let input_ids = &input_ids[..max_len];
    let attention_mask = &attention_mask[..max_len];
    let seq_len = max_len;

    let offsets = encoding.get_offsets();
    let offsets = &offsets[..seq_len];

    let token_byte_starts: Vec<usize> = offsets.iter().map(|o| o.0).collect();
    let token_byte_ends: Vec<usize> = offsets.iter().map(|o| o.1).collect();

    // 2. Build ONNX inputs and run inference.
    let input_ids_arr = ndarray::Array2::from_shape_vec((1, seq_len), input_ids.to_vec())
        .context("reshape input_ids")?;
    let attention_mask_arr = ndarray::Array2::from_shape_vec((1, seq_len), attention_mask.to_vec())
        .context("reshape attention_mask")?;

    let has_token_type_ids = cfg.onnx_inputs.contains(&"token_type_ids".to_string());
    let token_type_ids_arr = if has_token_type_ids {
        Some(
            ndarray::Array2::from_shape_vec((1, seq_len), vec![0i64; seq_len])
                .context("reshape token_type_ids")?,
        )
    } else {
        None
    };

    let outputs = if let Some(tti) = token_type_ids_arr {
        session
            .run(ort::inputs![
                "input_ids" => ort::value::Tensor::from_array(input_ids_arr).context("input_ids tensor")?,
                "attention_mask" => ort::value::Tensor::from_array(attention_mask_arr).context("attention_mask tensor")?,
                "token_type_ids" => ort::value::Tensor::from_array(tti).context("token_type_ids tensor")?,
            ])
            .context("ONNX inference failed")?
    } else {
        session
            .run(ort::inputs![
                "input_ids" => ort::value::Tensor::from_array(input_ids_arr).context("input_ids tensor")?,
                "attention_mask" => ort::value::Tensor::from_array(attention_mask_arr).context("attention_mask tensor")?,
            ])
            .context("ONNX inference failed")?
    };

    // 3. Extract logits [1, seq, num_labels].
    let output_name: &str = cfg
        .onnx_outputs
        .first()
        .map(String::as_str)
        .unwrap_or("logits");
    let (logits_shape, logits_data) = outputs[output_name]
        .try_extract_tensor::<f32>()
        .with_context(|| format!("extract logits from output '{}'", output_name))?;

    let num_labels = logits_shape.get(2).copied().unwrap_or(1) as usize;
    let actual_seq = logits_shape.get(1).copied().unwrap_or(0) as usize;

    // 4. Softmax per token → argmax label + score.
    let id2label = cfg.id2label_map();

    let (token_label_ids, token_scores) = {
        let mut ids: Vec<usize> = Vec::with_capacity(actual_seq);
        let mut scores: Vec<f32> = Vec::with_capacity(actual_seq);
        for t in 0..actual_seq {
            let offset = t * num_labels;
            // Softmax over classes for this token.
            let max_logit = logits_data[offset..offset + num_labels]
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f32 = logits_data[offset..offset + num_labels]
                .iter()
                .map(|&x| (x - max_logit).exp())
                .sum();
            let probs: Vec<f32> = logits_data[offset..offset + num_labels]
                .iter()
                .map(|&x| (x - max_logit).exp() / sum)
                .collect();
            let (best_idx, best_score) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));
            ids.push(best_idx);
            scores.push(*best_score);
        }
        (ids, scores)
    };

    // 5. BIO decode → entity spans.
    let mut entities = decode_bio(
        &token_label_ids,
        &token_byte_starts,
        &token_byte_ends,
        &token_scores,
        &id2label,
        threshold,
        labels,
    );

    // 6. Sort by start, truncate to max_spans.
    entities.sort_by_key(|(s, _, _, _)| *s);
    entities.truncate(max_spans);

    let result_entities: Vec<Entity> = entities
        .into_iter()
        .map(|(start, end, entity_type, score)| Entity {
            start,
            end,
            entity_type,
            score,
        })
        .collect();

    Ok(Response {
        entities: result_entities,
    })
}

// ---------------------------------------------------------------------------
// Long-text windowing (§6.11.6)
// ---------------------------------------------------------------------------

/// Run inference on overlapping windows to handle texts longer than max_len.
/// Merges spans by byte range to de-duplicate entities split across windows.
fn run_inference_windowed(
    text: &str,
    labels: &[String],
    session: &mut Session,
    tokenizer: &Tokenizer,
    cfg: &ModelConfig,
    threshold: f32,
    max_spans: usize,
) -> Result<Response> {
    // Quick check: if text fits in one window, just run it.
    // Tokenize once to see how many tokens it takes.
    let full_enc = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
    let full_len = full_enc.len();

    if full_len <= cfg.max_len {
        return run_inference(text, labels, session, tokenizer, cfg, threshold, max_spans);
    }

    // Overlapping windows.
    let stride = cfg.max_len.saturating_sub(64);
    let mut all_entities: Vec<(usize, usize, String, f32)> = Vec::new();

    let mut byte_pos = 0usize;
    while byte_pos < text.len() {
        // Take a window starting at byte_pos.
        let window_text = &text[byte_pos..];
        let resp = run_inference(
            window_text,
            labels,
            session,
            tokenizer,
            cfg,
            threshold,
            max_spans,
        )?;

        for ent in resp.entities {
            all_entities.push((
                byte_pos + ent.start,
                byte_pos + ent.end,
                ent.entity_type,
                ent.score,
            ));
        }

        // Advance by stride bytes.  Find a reasonable byte boundary.
        let advance = std::cmp::min(stride, text.len() - byte_pos);
        if advance == 0 {
            break;
        }
        // Try to advance to a space boundary for cleaner windows.
        let mut actual_advance = advance;
        // Walk back from byte_pos + advance to find a space.
        let target_byte = (byte_pos + advance).min(text.len());
        if target_byte < text.len() {
            // Search backward from target for space/newline.
            let range_start = byte_pos + advance / 2;
            if let Some(pos) = text[range_start..target_byte].rfind(|c: char| c.is_whitespace()) {
                actual_advance = (range_start + pos + 1) - byte_pos;
            }
        }
        if actual_advance == 0 {
            actual_advance = advance;
        }
        byte_pos += actual_advance;
    }

    // Deduplicate overlapping spans (keep higher score).
    all_entities.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    let mut deduped: Vec<(usize, usize, String, f32)> = Vec::new();
    for ent in all_entities {
        let overlaps = deduped.iter().any(|(s, e, _, _)| ent.0 < *e && *s < ent.1);
        if !overlaps {
            deduped.push(ent);
        }
    }
    deduped.sort_by_key(|(s, _, _, _)| *s);
    deduped.truncate(max_spans);

    Ok(Response {
        entities: deduped
            .into_iter()
            .map(|(start, end, entity_type, score)| Entity {
                start,
                end,
                entity_type,
                score,
            })
            .collect(),
    })
}

// ---------------------------------------------------------------------------
// Main — NDJSON loop
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load model config.
    let cfg_json = std::fs::read_to_string(&cli.config).context("read model config")?;
    let cfg: ModelConfig = serde_json::from_str(&cfg_json).context("parse model config")?;

    // Assert model_type so we don't silently load a GLiNER artifact.
    if cfg.model_type != "token-classifier-v1" {
        anyhow::bail!(
            "Expected model_type='token-classifier-v1', got '{}'. \
             Refusing to load incompatible model. \
             Re-run 'cloak init' to download the correct model.",
            cfg.model_type
        );
    }

    // Load ONNX model.
    let mut session = Session::builder()?
        .commit_from_file(&cli.model)
        .context("failed to load ONNX model")?;

    // Load tokenizer.
    let tokenizer = Tokenizer::from_file(&cli.tokenizer)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    use std::io::{BufRead, BufReader, Write};

    let stdin = std::io::stdin();
    let reader = BufReader::new(stdin.lock());
    let stdout = std::io::stdout();
    let mut writer = stdout.lock();

    for line in reader.lines() {
        let line = line.context("read stdin")?;
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let request: Request = serde_json::from_str(&line).context("parse JSON request")?;

        let response = run_inference_windowed(
            &request.text,
            &request.labels,
            &mut session,
            &tokenizer,
            &cfg,
            cli.threshold,
            cli.max_spans,
        )?;

        let json = serde_json::to_string(&response).context("serialize response")?;
        writeln!(writer, "{json}").context("write stdout")?;
        writer.flush().context("flush stdout")?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_bio_simple() {
        let id2label: HashMap<usize, String> = [
            (0, "O".to_string()),
            (1, "B-NAME".to_string()),
            (2, "I-NAME".to_string()),
            (3, "B-ADDRESS".to_string()),
            (4, "I-ADDRESS".to_string()),
            (5, "B-USERNAME".to_string()),
            (6, "I-USERNAME".to_string()),
        ]
        .into_iter()
        .collect();

        let token_ids = vec![0, 1, 2, 0, 3, 4, 0];
        let byte_starts = vec![0, 0, 5, 11, 12, 16, 22];
        let byte_ends = vec![0, 5, 11, 12, 16, 22, 23];
        let scores = vec![0.0, 0.95, 0.90, 0.0, 0.88, 0.85, 0.0];
        let allow = vec!["NAME".to_string(), "ADDRESS".to_string()];

        let result = decode_bio(
            &token_ids,
            &byte_starts,
            &byte_ends,
            &scores,
            &id2label,
            0.5,
            &allow,
        );

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (0, 11, "NAME".to_string(), 0.90));
        assert_eq!(result[1], (12, 22, "ADDRESS".to_string(), 0.85));
    }

    #[test]
    fn test_decode_bio_filter_allow_list() {
        let id2label: HashMap<usize, String> = [
            (0, "O".to_string()),
            (1, "B-NAME".to_string()),
            (2, "I-NAME".to_string()),
            (3, "B-USERNAME".to_string()),
            (4, "I-USERNAME".to_string()),
        ]
        .into_iter()
        .collect();

        // "B-NAME I-NAME B-USERNAME I-USERNAME" but USERNAME not in allow-list.
        let token_ids = vec![0, 1, 2, 3, 4, 0];
        let byte_starts = vec![0, 0, 3, 7, 12, 19];
        let byte_ends = vec![0, 3, 7, 12, 19, 20];
        let scores = vec![0.0, 0.95, 0.90, 0.88, 0.85, 0.0];
        let allow = vec!["NAME".to_string()]; // USERNAME not allowed

        let result = decode_bio(
            &token_ids,
            &byte_starts,
            &byte_ends,
            &scores,
            &id2label,
            0.5,
            &allow,
        );

        assert_eq!(result.len(), 1, "USERNAME should be filtered out");
        assert_eq!(result[0], (0, 7, "NAME".to_string(), 0.90));
    }

    #[test]
    fn test_decode_bio_score_min() {
        let id2label: HashMap<usize, String> = [
            (0, "O".to_string()),
            (1, "B-NAME".to_string()),
            (2, "I-NAME".to_string()),
        ]
        .into_iter()
        .collect();

        let token_ids = vec![1, 2, 2]; // B-NAME, I-NAME, I-NAME
        let byte_starts = vec![0, 3, 8];
        let byte_ends = vec![3, 8, 13];
        let scores = vec![0.99, 0.70, 0.95];
        let allow = vec!["NAME".to_string()];

        let result = decode_bio(
            &token_ids,
            &byte_starts,
            &byte_ends,
            &scores,
            &id2label,
            0.5,
            &allow,
        );

        assert_eq!(result.len(), 1);
        // Score should be min(0.99, 0.70, 0.95) = 0.70
        assert!(
            (result[0].3 - 0.70).abs() < 0.001,
            "expected 0.70, got {}",
            result[0].3
        );
    }

    #[test]
    fn test_decode_bio_threshold_filters() {
        let id2label: HashMap<usize, String> = [
            (0, "O".to_string()),
            (1, "B-NAME".to_string()),
            (2, "I-NAME".to_string()),
        ]
        .into_iter()
        .collect();

        let token_ids = vec![1, 2]; // B-NAME (score 0.6), I-NAME (score 0.3)
        let byte_starts = vec![0, 5];
        let byte_ends = vec![5, 10];
        let scores = vec![0.6, 0.3];
        let allow = vec!["NAME".to_string()];

        // threshold 0.5: I-NAME at 0.3 is below → treated as O, span closes
        let result = decode_bio(
            &token_ids,
            &byte_starts,
            &byte_ends,
            &scores,
            &id2label,
            0.5,
            &allow,
        );

        assert_eq!(result.len(), 1, "I-NAME below threshold should close span");
        assert_eq!(result[0].0, 0); // byte_start of first token
        assert_eq!(result[0].1, 5); // byte_end of first token only

        // threshold 0.2: both above threshold, span should cover both tokens
        let result2 = decode_bio(
            &token_ids,
            &byte_starts,
            &byte_ends,
            &scores,
            &id2label,
            0.2,
            &allow,
        );
        assert_eq!(result2.len(), 1);
        assert_eq!(result2[0].0, 0);
        assert_eq!(result2[0].1, 10); // covers both tokens
    }
}
