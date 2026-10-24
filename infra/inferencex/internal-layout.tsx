import '@/lib/runtime/polyfills';
import './globals.css';

import type { Metadata } from 'next';
import { DM_Sans } from 'next/font/google';
import { SITE_TITLE, SITE_URL } from '@semianalysisai/inferencex-constants';
import { ThemeProvider } from '@/components/ui/theme-provider';
import { QueryProvider } from '@/providers/query-provider';
import { PostHogProvider } from '@/providers/posthog-provider';

const font = DM_Sans({ subsets: ['latin'], display: 'optional', variable: '--font-dm-sans' });

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: SITE_TITLE,
  robots: { index: false, follow: false },
};

export const viewport = { width: 'device-width', initialScale: 1, maximumScale: 5 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${font.variable} antialiased relative min-h-screen flex flex-col`}>
        <PostHogProvider>
          <QueryProvider>
            <ThemeProvider
              attribute="class"
              defaultTheme="dark"
              enableSystem
              disableTransitionOnChange
            >
              {children}
            </ThemeProvider>
          </QueryProvider>
        </PostHogProvider>
      </body>
    </html>
  );
}
