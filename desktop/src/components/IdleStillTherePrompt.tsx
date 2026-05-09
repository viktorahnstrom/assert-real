interface IdleStillTherePromptProps {
  open: boolean;
  onContinue: () => void;
  onDiscard: () => void;
}

// Shown when useSectionTimer detects the participant has been idle past
// the configured threshold. "Yes, continue" resumes the timer; "I'm done"
// marks this section's time as discarded but lets the participant keep
// going through the study.
export function IdleStillTherePrompt({ open, onContinue, onDiscard }: IdleStillTherePromptProps) {
  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="idle-prompt-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-xade-charcoal/40 px-6 backdrop-blur-sm"
    >
      <div className="w-full max-w-sm rounded-2xl border border-xade-charcoal/6 bg-white px-7 py-7 shadow-xl">
        <h2 id="idle-prompt-title" className="text-lg font-semibold text-xade-charcoal">
          Are you still there?
        </h2>
        <p className="mt-2 text-sm leading-relaxed text-xade-charcoal/60">
          We paused the timer because you have been inactive. If you are still here, click continue
          and we will pick up where you left off.
        </p>

        <div className="mt-6 flex flex-col gap-2">
          <button
            onClick={onContinue}
            className="flex w-full items-center justify-center rounded-lg bg-xade-blue px-4 py-3 text-sm font-medium text-white transition-colors hover:bg-xade-blue-dark"
          >
            Yes, continue
          </button>
          <button
            onClick={onDiscard}
            className="flex w-full items-center justify-center rounded-lg border border-xade-charcoal/15 bg-white px-4 py-3 text-sm font-medium text-xade-charcoal/70 transition-colors hover:bg-xade-charcoal/3"
          >
            I&apos;m done — discard this section&apos;s time
          </button>
        </div>
      </div>
    </div>
  );
}
