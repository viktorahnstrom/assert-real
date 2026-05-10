interface IdleStillTherePromptProps {
  open: boolean;
  onContinue: () => void;
  onDiscard: () => void;
}

// Shown when useSectionTimer detects the participant has been idle past
// the configured threshold. Both buttons let the participant continue
// the study; the second one tells us their time on this section was not
// active engagement, so analysis can exclude it.
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
          We paused the timer because there was no activity. Click continue when you are ready, or
          let us know you stepped away so we do not count this section&apos;s time.
        </p>

        <div className="mt-6 flex flex-col gap-2">
          <button
            onClick={onContinue}
            className="flex w-full items-center justify-center rounded-lg bg-xade-blue px-4 py-3 text-sm font-medium text-white transition-colors hover:bg-xade-blue-dark"
          >
            Yes, I am still here
          </button>
          <button
            onClick={onDiscard}
            className="flex w-full items-center justify-center rounded-lg border border-xade-charcoal/15 bg-white px-4 py-3 text-sm font-medium text-xade-charcoal/70 transition-colors hover:bg-xade-charcoal/3"
          >
            I stepped away briefly
          </button>
        </div>
      </div>
    </div>
  );
}
