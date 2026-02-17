import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electronAPI', {
  getAppVersion: (): Promise<string> => ipcRenderer.invoke('get-app-version'),
  platform: process.platform,
});

declare global {
  interface Window {
    electronAPI: {
      getAppVersion: () => Promise<string>;
      platform: NodeJS.Platform;
    };
  }
}
