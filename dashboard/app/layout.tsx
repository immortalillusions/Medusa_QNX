import "./globals.css";

export const metadata = { title: "Medusa · QNX demo", description: "Shoulder-surfer guard with a QNX real-time turret" };
export const viewport = { width: "device-width", initialScale: 1, themeColor: "#0e1116" };

export default function Layout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
