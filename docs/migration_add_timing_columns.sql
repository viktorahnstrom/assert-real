-- Migration: Add per-section timing columns to study_results (issue #118).
-- Run this in the Supabase SQL Editor.
--
-- Per-image timings live inside the existing JSONB arrays
-- (classification_records, explanation_answers) — no schema change is
-- needed for those. The top-level columns below let the analysis script
-- aggregate without re-parsing JSONB.

ALTER TABLE study_results
  ADD COLUMN IF NOT EXISTS classification_records JSONB,
  ADD COLUMN IF NOT EXISTS phase4_time_ms FLOAT8,
  ADD COLUMN IF NOT EXISTS total_time_ms FLOAT8,
  ADD COLUMN IF NOT EXISTS total_idle_discarded_ms FLOAT8;
