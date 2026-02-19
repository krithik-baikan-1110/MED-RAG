import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export interface MedicalImage {
  id: string;
  user_id: string;
  image_url: string;
  image_name: string;
  extracted_details: Record<string, any>;
  created_at: string;
}

export interface ChatMessage {
  id: string;
  user_id: string;
  message: string;
  response: string;
  image_id?: string;
  created_at: string;
}

export interface UserProfile {
  id: string;
  full_name: string;
  created_at: string;
}
