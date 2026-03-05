import { useState } from 'react';
import { Eye, EyeOff, Mail, Lock, User, ArrowRight, Loader2 } from 'lucide-react';
import { supabase } from '@/lib/supabase';

type AuthMode = 'login' | 'signup' | 'forgot';

interface AuthError {
  message: string;
}

function GoogleIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none">
      <path
        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"
        fill="#4285F4"
      />
      <path
        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
        fill="#34A853"
      />
      <path
        d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
        fill="#FBBC05"
      />
      <path
        d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
        fill="#EA4335"
      />
    </svg>
  );
}

function Divider() {
  return (
    <div className="relative my-6">
      <div className="absolute inset-0 flex items-center">
        <div className="w-full border-t border-xade-charcoal/10" />
      </div>
      <div className="relative flex justify-center text-xs">
        <span className="bg-white px-3 text-xade-charcoal/40">or continue with email</span>
      </div>
    </div>
  );
}

export default function AuthPage() {
  const [mode, setMode] = useState<AuthMode>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);
  const [error, setError] = useState<AuthError | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  function resetForm() {
    setEmail('');
    setPassword('');
    setDisplayName('');
    setError(null);
    setSuccess(null);
  }

  function switchMode(newMode: AuthMode) {
    resetForm();
    setMode(newMode);
  }

  async function handleGoogleLogin() {
    setGoogleLoading(true);
    setError(null);

    try {
      const { error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo: window.location.origin,
        },
      });

      if (error) throw error;
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Google login failed';
      setError({ message });
      setGoogleLoading(false);
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      if (mode === 'login') {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
      } else if (mode === 'signup') {
        const { error } = await supabase.auth.signUp({
          email,
          password,
          options: {
            data: { display_name: displayName || undefined },
          },
        });
        if (error) throw error;
        setSuccess('Account created! Check your email to verify, then log in.');
        setTimeout(() => switchMode('login'), 3000);
      } else if (mode === 'forgot') {
        const { error } = await supabase.auth.resetPasswordForEmail(email, {
          redirectTo: `${window.location.origin}/reset-password`,
        });
        if (error) throw error;
        setSuccess('Password reset email sent. Check your inbox.');
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Something went wrong';
      setError({ message });
    } finally {
      setLoading(false);
    }
  }

  const titles: Record<AuthMode, { heading: string; sub: string }> = {
    login: {
      heading: 'Welcome back',
      sub: 'Sign in to continue your deepfake analysis',
    },
    signup: {
      heading: 'Create account',
      sub: 'Start detecting deepfakes with AI explanations',
    },
    forgot: {
      heading: 'Reset password',
      sub: "Enter your email and we'll send a reset link",
    },
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-xade-cream p-6">
      {/* Background pattern */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -left-32 -top-32 h-96 w-96 rounded-full bg-xade-blue/4 blur-3xl" />
        <div className="absolute -bottom-48 -right-48 h-128 w-lg rounded-full bg-xade-blue/3 blur-3xl" />
      </div>

      <div className="relative w-full max-w-sm">
        {/* Logo */}
        <div className="mb-10 text-center">
          <h1 className="text-4xl font-bold tracking-tight text-xade-blue">XADE</h1>
          <p className="mt-1 text-[11px] font-medium uppercase tracking-[0.2em] text-xade-charcoal/40">
            Deepfake Detection
          </p>
        </div>

        {/* Card */}
        <div className="rounded-2xl border border-xade-charcoal/6 bg-white px-8 py-8 shadow-lg shadow-xade-charcoal/4">
          {/* Title */}
          <div className="mb-6">
            <h2 className="text-xl font-semibold text-xade-charcoal">{titles[mode].heading}</h2>
            <p className="mt-1 text-sm text-xade-charcoal/50">{titles[mode].sub}</p>
          </div>

          {/* Google button (not on forgot) */}
          {mode !== 'forgot' && (
            <>
              <button
                type="button"
                onClick={handleGoogleLogin}
                disabled={googleLoading}
                className="flex w-full items-center justify-center gap-3 rounded-lg border border-xade-charcoal/10 bg-white px-4 py-2.5 text-sm font-medium text-xade-charcoal transition-colors hover:bg-xade-charcoal/3 disabled:opacity-50"
              >
                {googleLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <GoogleIcon className="h-4.5 w-4.5" />
                )}
                Continue with Google
              </button>
              <Divider />
            </>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Display name (signup only) */}
            {mode === 'signup' && (
              <div>
                <label className="mb-1.5 block text-xs font-medium text-xade-charcoal/60">
                  Display name
                </label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-xade-charcoal/30" />
                  <input
                    type="text"
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                    placeholder="Your name"
                    className="w-full rounded-lg border border-xade-charcoal/10 bg-white py-2.5 pl-10 pr-4 text-sm text-xade-charcoal placeholder:text-xade-charcoal/30 focus:border-xade-blue/40 focus:outline-none focus:ring-2 focus:ring-xade-blue/10"
                  />
                </div>
              </div>
            )}

            {/* Email */}
            <div>
              <label className="mb-1.5 block text-xs font-medium text-xade-charcoal/60">
                Email
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-xade-charcoal/30" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  required
                  className="w-full rounded-lg border border-xade-charcoal/10 bg-white py-2.5 pl-10 pr-4 text-sm text-xade-charcoal placeholder:text-xade-charcoal/30 focus:border-xade-blue/40 focus:outline-none focus:ring-2 focus:ring-xade-blue/10"
                />
              </div>
            </div>

            {/* Password (not on forgot) */}
            {mode !== 'forgot' && (
              <div>
                <div className="mb-1.5 flex items-center justify-between">
                  <label className="text-xs font-medium text-xade-charcoal/60">Password</label>
                  {mode === 'login' && (
                    <button
                      type="button"
                      onClick={() => switchMode('forgot')}
                      className="text-xs text-xade-blue/70 hover:text-xade-blue"
                    >
                      Forgot password?
                    </button>
                  )}
                </div>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-xade-charcoal/30" />
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder={mode === 'signup' ? 'Min 6 characters' : '••••••••'}
                    required
                    minLength={mode === 'signup' ? 6 : undefined}
                    className="w-full rounded-lg border border-xade-charcoal/10 bg-white py-2.5 pl-10 pr-10 text-sm text-xade-charcoal placeholder:text-xade-charcoal/30 focus:border-xade-blue/40 focus:outline-none focus:ring-2 focus:ring-xade-blue/10"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword((prev) => !prev)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-xade-charcoal/30 hover:text-xade-charcoal/60"
                  >
                    {showPassword ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </button>
                </div>
              </div>
            )}

            {/* Error message */}
            {error && (
              <div className="rounded-lg bg-red-50 px-4 py-2.5 text-xs text-red-600">
                {error.message}
              </div>
            )}

            {/* Success message */}
            {success && (
              <div className="rounded-lg bg-green-50 px-4 py-2.5 text-xs text-green-600">
                {success}
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              disabled={loading}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-xade-blue px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-xade-blue-dark disabled:opacity-50"
            >
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <>
                  {mode === 'login' && 'Sign in'}
                  {mode === 'signup' && 'Create account'}
                  {mode === 'forgot' && 'Send reset link'}
                  <ArrowRight className="h-4 w-4" />
                </>
              )}
            </button>
          </form>
        </div>

        {/* Bottom toggle */}
        <div className="mt-6 text-center text-sm text-xade-charcoal/50">
          {mode === 'login' && (
            <>
              Don&apos;t have an account?{' '}
              <button
                onClick={() => switchMode('signup')}
                className="font-medium text-xade-blue hover:underline"
              >
                Sign up
              </button>
            </>
          )}
          {mode === 'signup' && (
            <>
              Already have an account?{' '}
              <button
                onClick={() => switchMode('login')}
                className="font-medium text-xade-blue hover:underline"
              >
                Sign in
              </button>
            </>
          )}
          {mode === 'forgot' && (
            <>
              Remember your password?{' '}
              <button
                onClick={() => switchMode('login')}
                className="font-medium text-xade-blue hover:underline"
              >
                Back to sign in
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}