import { BarChart3, HelpCircle, History, Upload, User, MoreVertical } from 'lucide-react';
import {
  Button,
  SidebarProvider,
  Sidebar,
  SidebarHeader,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarItem,
  SidebarTrigger,
  SidebarInset,
  useSidebar,
} from '@/components/ui';

function SidebarLogo() {
  const { isCollapsed } = useSidebar();

  return (
    <div className="flex items-center gap-2">
      {!isCollapsed && <span className="text-lg font-semibold text-xade-blue">XADE</span>}
    </div>
  );
}

function UserProfile() {
  const { isCollapsed } = useSidebar();

  return (
    <div className="flex items-center gap-3">
      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-xade-charcoal/10">
        <User className="h-4 w-4 text-xade-charcoal/70" />
      </div>
      {!isCollapsed && (
        <>
          <span className="flex-1 text-sm font-medium text-xade-charcoal">John Doe</span>
          <button className="text-xade-charcoal/50 hover:text-xade-charcoal">
            <MoreVertical className="h-4 w-4" />
          </button>
        </>
      )}
    </div>
  );
}

function AppSidebar() {
  return (
    <Sidebar>
      <SidebarHeader>
        <SidebarLogo />
        <SidebarTrigger />
      </SidebarHeader>

      <SidebarContent>
        {/* Main Navigation */}
        <SidebarGroup>
          <SidebarItem icon={<BarChart3 />}>Statistics</SidebarItem>
          <SidebarItem icon={<HelpCircle />}>Support</SidebarItem>
          <SidebarItem icon={<History />}>History</SidebarItem>
        </SidebarGroup>

        {/* Recent Section */}
        <SidebarGroup label="Recent">
          <SidebarItem>Lorem</SidebarItem>
          <SidebarItem>Lorem</SidebarItem>
          <SidebarItem>Lorem</SidebarItem>
          <SidebarItem>Lorem</SidebarItem>
          <SidebarItem>Lorem</SidebarItem>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <UserProfile />
      </SidebarFooter>
    </Sidebar>
  );
}

function MainContent() {
  return (
    <SidebarInset>
      <div className="flex min-h-screen flex-col items-center justify-center p-8">
        {/* XADE Logo */}
        <div className="mb-8 text-center">
          <div className="mb-2 h-px w-48 bg-xade-charcoal/20" />
          <h1 className="text-7xl font-bold tracking-tight text-xade-blue">XADE</h1>
          <div className="mt-2 h-px w-48 bg-xade-charcoal/20" />
        </div>

        {/* Upload Box */}
        <div className="w-full max-w-md">
          <div className="flex flex-col items-center justify-center rounded-lg border-2 border-xade-blue/30 bg-white p-12 transition-colors hover:border-xade-blue/50">
            <Upload className="mb-4 h-16 w-16 text-xade-blue/50" strokeWidth={1} />
            <p className="mb-1 text-lg font-medium text-xade-charcoal">
              Drag and drop or click here
            </p>
            <p className="text-sm text-xade-charcoal/50">to upload your image (max 2mb)</p>
          </div>

          {/* Submit Button */}
          <div className="mt-6 flex justify-center">
            <Button variant="outline" className="min-w-32">
              Submit
            </Button>
          </div>
        </div>
      </div>
    </SidebarInset>
  );
}

function App() {
  return (
    <SidebarProvider>
      <AppSidebar />
      <MainContent />
    </SidebarProvider>
  );
}

export default App;
