import NextAuth from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";

const API = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

const handler = NextAuth({
  secret: process.env.NEXTAUTH_SECRET || "dicv-local-development-secret-key-12345",
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        username: { label: "Username", type: "text", placeholder: "Username" },
        password: { label: "Password", type: "password" }
      },
      async authorize(credentials) {
        if (!credentials?.username || !credentials?.password) {
          return null;
        }

        try {
          const params = new URLSearchParams();
          params.append('username', credentials.username);
          params.append('password', credentials.password);

          const res = await fetch(`${API}/api/auth/login`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: params.toString(),
          });

          const data = await res.json();
          console.log("Login response status:", res.status);
          console.log("Login response data:", data);

          if (res.ok && data.access_token) {
            return {
              id: data.user_id.toString(),
              name: data.username,
              accessToken: data.access_token,
            };
          }
          
          return null;
        } catch (e) {
          console.error("Login fetch error:", e);
          return null;
        }
      }
    })
  ],
  session: {
    strategy: "jwt",
    maxAge: 7 * 24 * 60 * 60, // 7 days matching backend
  },
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.id = user.id;
        token.accessToken = (user as any).accessToken;
      }
      return token;
    },
    async session({ session, token }) {
      if (token && session.user) {
        (session.user as any).id = token.id as string;
        (session as any).accessToken = token.accessToken;
      }
      return session;
    }
  },
  pages: {
    signIn: '/login',
  },
});

export { handler as GET, handler as POST };
