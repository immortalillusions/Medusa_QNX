import "./globals.css";

export const metadata = { title: "Medusa", description: "Local QNX preview" };

export default function Layout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}

