import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import type { User } from '@/types'

interface AuthStore {
  user: User | null
  token: string | null
  _hydrated: boolean
  setAuth: (user: User, token: string) => void
  logout: () => void
  setHydrated: (v: boolean) => void
}

export const useAuthStore = create<AuthStore>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      _hydrated: false,
      setHydrated: (v) => set({ _hydrated: v }),
      setAuth: (user, token) => set({ user, token }),
      logout: () => set({ user: null, token: null }),
    }),
    {
      name: 'papermind-auth',
      storage: createJSONStorage(() => localStorage),
      partialize: (s) => ({ user: s.user, token: s.token }),
      onRehydrateStorage: () => (state) => {
        state?.setHydrated(true)
      },
    }
  )
)
