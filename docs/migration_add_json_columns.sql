-- Migration: Add JSON columns for explanation and evidence regions
-- Run this in the Supabase SQL Editor

ALTER TABLE analyses ADD COLUMN IF NOT EXISTS explanation_json JSONB;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS evidence_regions_json JSONB;
