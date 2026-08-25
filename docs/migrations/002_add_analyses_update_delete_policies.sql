-- Migration 002: Add missing RLS policies for analyses table.
--
-- The analyses table had SELECT and INSERT policies but no UPDATE or DELETE.
-- With the auth fix (commit 4f0f51a), all Supabase calls now run as the
-- authenticated user instead of the service role key, so these policies are
-- required for the product flow to work:
--   - UPDATE: create_analysis patches status from "processing" to "completed"
--   - DELETE: delete_analysis removes a user's own analysis
--
-- Run this in the Supabase SQL Editor.

-- Users can update their own analyses (status, results, error_message).
CREATE POLICY "Users can update own analyses" ON analyses
    FOR UPDATE USING (auth.uid() = user_id);

-- Users can delete their own analyses.
CREATE POLICY "Users can delete own analyses" ON analyses
    FOR DELETE USING (auth.uid() = user_id);

-- Note: api_logs INSERT is intentionally omitted. That table is written by
-- server-side instrumentation using the service role key, not by end users.
-- Adding an anon/user INSERT policy would let anyone fabricate log entries.
