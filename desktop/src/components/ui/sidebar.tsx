import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { PanelLeft } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from './button';

// Sidebar Context
type SidebarContextValue = {
  isCollapsed: boolean;
  toggle: () => void;
};

const SidebarContext = React.createContext<SidebarContextValue | null>(null);

export function useSidebar() {
  const context = React.useContext(SidebarContext);
  if (!context) {
    throw new Error('useSidebar must be used within a SidebarProvider');
  }
  return context;
}

// Sidebar Provider
interface SidebarProviderProps {
  children: React.ReactNode;
  defaultCollapsed?: boolean;
}

export function SidebarProvider({ children, defaultCollapsed = false }: SidebarProviderProps) {
  const [isCollapsed, setIsCollapsed] = React.useState(defaultCollapsed);

  const toggle = React.useCallback(() => {
    setIsCollapsed((prev) => !prev);
  }, []);

  return (
    <SidebarContext.Provider value={{ isCollapsed, toggle }}>
      <div className="flex min-h-screen w-full">{children}</div>
    </SidebarContext.Provider>
  );
}

// Sidebar
const sidebarVariants = cva(
  'flex h-screen flex-col border-r border-xade-charcoal/10 bg-xade-cream transition-all duration-300 ease-in-out',
  {
    variants: {
      collapsed: {
        true: 'w-16',
        false: 'w-64',
      },
    },
    defaultVariants: {
      collapsed: false,
    },
  }
);

interface SidebarProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

export function Sidebar({ children, className, ...props }: SidebarProps) {
  const { isCollapsed } = useSidebar();

  return (
    <aside className={cn(sidebarVariants({ collapsed: isCollapsed }), className)} {...props}>
      {children}
    </aside>
  );
}

// Sidebar Header
export function SidebarHeader({
  children,
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('flex items-center justify-between p-4', className)} {...props}>
      {children}
    </div>
  );
}

// Sidebar Content
export function SidebarContent({
  children,
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('flex-1 overflow-y-auto px-3 py-2', className)} {...props}>
      {children}
    </div>
  );
}

// Sidebar Footer
export function SidebarFooter({
  children,
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('mt-auto border-t border-xade-charcoal/10 p-4', className)} {...props}>
      {children}
    </div>
  );
}

// Sidebar Group
interface SidebarGroupProps extends React.HTMLAttributes<HTMLDivElement> {
  label?: string;
}

export function SidebarGroup({ children, label, className, ...props }: SidebarGroupProps) {
  const { isCollapsed } = useSidebar();

  return (
    <div className={cn('py-2', className)} {...props}>
      {label && !isCollapsed && (
        <p className="mb-2 px-3 text-xs font-medium uppercase tracking-wider text-xade-charcoal/50">
          {label}
        </p>
      )}
      <nav className="space-y-1">{children}</nav>
    </div>
  );
}

// Sidebar Item
interface SidebarItemProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  icon?: React.ReactNode;
  isActive?: boolean;
}

export function SidebarItem({ children, icon, isActive, className, ...props }: SidebarItemProps) {
  const { isCollapsed } = useSidebar();

  return (
    <button
      className={cn(
        'flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors',
        'hover:bg-xade-charcoal/5 hover:text-xade-charcoal',
        isActive ? 'bg-xade-blue/10 text-xade-blue' : 'text-xade-charcoal/70',
        isCollapsed && 'justify-center px-2',
        className
      )}
      {...props}
    >
      {icon && <span className="h-5 w-5 shrink-0">{icon}</span>}
      {!isCollapsed && <span>{children}</span>}
    </button>
  );
}

// Sidebar Trigger
export function SidebarTrigger({
  className,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const { toggle } = useSidebar();

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={toggle}
      className={cn('h-8 w-8', className)}
      {...props}
    >
      <PanelLeft className="h-4 w-4" />
    </Button>
  );
}

// Sidebar Inset (main content area)
export function SidebarInset({
  children,
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <main className={cn('flex-1 overflow-auto bg-white', className)} {...props}>
      {children}
    </main>
  );
}
