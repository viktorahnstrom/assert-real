import { useCallback, useEffect, useRef, useState } from 'react';

// Idle thresholds per phase (configurable via the optional argument).
// Phase 1 and Phase 3 are rapid binary classification, so 2 min is plenty.
// Phase 2 (denser reading) and Phase 4 (survey writing) get 3 min.
export const IDLE_THRESHOLD_PHASE_1_MS = 2 * 60 * 1000;
export const IDLE_THRESHOLD_PHASE_2_MS = 3 * 60 * 1000;
export const IDLE_THRESHOLD_PHASE_4_MS = 3 * 60 * 1000;

const ACTIVITY_EVENTS = ['mousemove', 'click', 'keydown', 'touchstart', 'scroll'] as const;
const IDLE_TICK_MS = 1000;

export interface SectionTimerSnapshot {
  timeMs: number;
  idleDiscarded: boolean;
  idlePausedMs: number;
}

export interface UseSectionTimerReturn {
  isIdleModalOpen: boolean;
  confirmStillThere: () => void;
  markDiscarded: () => void;
  finalize: () => SectionTimerSnapshot;
}

// Tracks active engagement time for one section. Pauses when the user is
// idle for `idleThresholdMs` (and shows a modal) or when the tab is hidden.
// Resets whenever `sectionKey` changes — pass a stable key per section
// (e.g. the image id, or 'survey') so the timer restarts cleanly between
// sections without unmounting the consumer.
export function useSectionTimer(
  sectionKey: string | number,
  idleThresholdMs: number = IDLE_THRESHOLD_PHASE_1_MS
): UseSectionTimerReturn {
  const [isIdleModalOpen, setIsIdleModalOpen] = useState(false);

  const accumulatedMsRef = useRef(0);
  const idlePausedMsRef = useRef(0);
  const lastResumeAtRef = useRef(Date.now());
  const lastActivityAtRef = useRef(Date.now());
  const pausedRef = useRef(false);
  const pauseStartRef = useRef(0);
  const idleDiscardedRef = useRef(false);
  const finalizedRef = useRef(false);

  // Reset whenever the section identity changes.
  useEffect(() => {
    const now = Date.now();
    accumulatedMsRef.current = 0;
    idlePausedMsRef.current = 0;
    lastResumeAtRef.current = now;
    lastActivityAtRef.current = now;
    pausedRef.current = false;
    pauseStartRef.current = 0;
    idleDiscardedRef.current = false;
    finalizedRef.current = false;
    setIsIdleModalOpen(false);
  }, [sectionKey]);

  const pause = useCallback((dueToIdle: boolean) => {
    if (pausedRef.current || finalizedRef.current) return;
    const now = Date.now();
    accumulatedMsRef.current += now - lastResumeAtRef.current;
    pausedRef.current = true;
    pauseStartRef.current = now;
    if (dueToIdle) setIsIdleModalOpen(true);
  }, []);

  const resume = useCallback(() => {
    if (!pausedRef.current || finalizedRef.current) return;
    const now = Date.now();
    idlePausedMsRef.current += now - pauseStartRef.current;
    lastResumeAtRef.current = now;
    lastActivityAtRef.current = now;
    pausedRef.current = false;
  }, []);

  // Idle-threshold polling.
  useEffect(() => {
    const id = window.setInterval(() => {
      if (pausedRef.current || finalizedRef.current) return;
      if (Date.now() - lastActivityAtRef.current >= idleThresholdMs) {
        pause(true);
      }
    }, IDLE_TICK_MS);
    return () => window.clearInterval(id);
  }, [idleThresholdMs, pause]);

  // Activity events that reset the idle countdown.
  useEffect(() => {
    const onActivity = () => {
      lastActivityAtRef.current = Date.now();
    };
    for (const ev of ACTIVITY_EVENTS) {
      window.addEventListener(ev, onActivity, { passive: true });
    }
    return () => {
      for (const ev of ACTIVITY_EVENTS) {
        window.removeEventListener(ev, onActivity);
      }
    };
  }, []);

  // Tab visibility: pause silently when hidden, resume when visible
  // (unless an idle prompt is awaiting user input).
  useEffect(() => {
    const onVisibility = () => {
      if (document.hidden) {
        pause(false);
      } else if (!isIdleModalOpen) {
        resume();
      }
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => document.removeEventListener('visibilitychange', onVisibility);
  }, [pause, resume, isIdleModalOpen]);

  const confirmStillThere = useCallback(() => {
    setIsIdleModalOpen(false);
    resume();
  }, [resume]);

  const markDiscarded = useCallback(() => {
    idleDiscardedRef.current = true;
    setIsIdleModalOpen(false);
    resume();
  }, [resume]);

  const finalize = useCallback((): SectionTimerSnapshot => {
    if (!finalizedRef.current) {
      if (!pausedRef.current) {
        accumulatedMsRef.current += Date.now() - lastResumeAtRef.current;
        pausedRef.current = true;
      }
      finalizedRef.current = true;
    }
    return {
      timeMs: accumulatedMsRef.current,
      idleDiscarded: idleDiscardedRef.current,
      idlePausedMs: idlePausedMsRef.current,
    };
  }, []);

  return { isIdleModalOpen, confirmStillThere, markDiscarded, finalize };
}
