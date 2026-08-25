-- Migration 001: Create study_results table for anonymous user study data.
--
-- This table stores anonymised participant results from the deepfake
-- detection user study.  It is written by the frontend directly using
-- the Supabase anon key (no backend involvement).
--
-- SECURITY NOTE: The anon key is embedded in the JS bundle that every
-- visitor receives.  Only an INSERT policy exists for anon.  There is
-- intentionally NO SELECT policy for anon.  Adding one would expose
-- every participant's raw study data to any visitor, since the anon key
-- is public.  Read access should only happen via the service role key
-- (Supabase dashboard or the show_study_results.py admin script).
--
-- Previously documented inline in docs/deploy.md.  Moved here so all
-- schema lives in one place.
--
-- Run this in the Supabase SQL Editor.

CREATE TABLE IF NOT EXISTS study_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    participant_id TEXT NOT NULL,
    self_confidence_rating INT,
    baseline_accuracy FLOAT,
    total_images INT,
    correct_count INT,
    incorrect_count INT,
    classification_records JSONB,
    explanation_answers JSONB,
    retest_answers JSONB,
    trust_rating INT,
    willingness_to_use TEXT,
    explanations_helped_in_retest INT,
    comments TEXT,
    phase4_time_ms FLOAT8,
    total_time_ms FLOAT8,
    total_idle_discarded_ms FLOAT8,
    completed_at TIMESTAMPTZ,
    saved_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE study_results ENABLE ROW LEVEL SECURITY;

-- Anonymous visitors can insert their own study results.
-- No SELECT/UPDATE/DELETE policy for anon — see security note above.
CREATE POLICY "anon can insert" ON study_results
    FOR INSERT TO anon WITH CHECK (true);
