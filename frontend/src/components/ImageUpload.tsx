import { useMemo, useState } from 'react';
import { Upload, Image as ImageIcon, X, CheckCircle, Loader2 } from 'lucide-react';
import { supabase, type MedicalImage } from '../lib/supabase';
import { useAuth } from '../contexts/AuthContext';
import { API_BASE_URL } from '../lib/api';

interface ImageUploadProps {
  onImageUploaded: (image: MedicalImage) => void;
}

async function compressImage(file: File): Promise<File> {
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error ?? new Error('Unable to read image'));
    reader.readAsDataURL(file);
  });

  const image = await new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Unable to load image for compression'));
    img.src = dataUrl;
  });

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Canvas context unavailable for compression');
  }

  const maxDimension = 1600;
  let { width, height } = image;
  if (width > height && width > maxDimension) {
    const ratio = maxDimension / width;
    width = maxDimension;
    height = Math.round(height * ratio);
  } else if (height > maxDimension) {
    const ratio = maxDimension / height;
    height = maxDimension;
    width = Math.round(width * ratio);
  }

  canvas.width = width;
  canvas.height = height;
  ctx.drawImage(image, 0, 0, width, height);

  const compressedBlob: Blob = await new Promise((resolve, reject) => {
    canvas.toBlob(
      blob => (blob ? resolve(blob) : reject(new Error('Failed to generate compressed image'))),
      'image/jpeg',
      0.85,
    );
  });

  const baseName = file.name.replace(/\.[^/.]+$/, '') || 'image';
  return new File([compressedBlob], `${baseName}-optimized.jpg`, { type: 'image/jpeg' });
}

export default function ImageUpload({ onImageUploaded }: ImageUploadProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [compressedPreview, setCompressedPreview] = useState<string>('');
  const [compressedInfo, setCompressedInfo] = useState<{ original: number; compressed: number } | null>(null);
  const [preview, setPreview] = useState<string>('');
  const [uploading, setUploading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const { user } = useAuth();

  const humanReadableSize = useMemo(() => {
    if (!compressedInfo) return null;
    const format = (bytes: number) => {
      if (bytes < 1024) return `${bytes.toFixed(0)} B`;
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
      return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    };
    return {
      original: format(compressedInfo.original),
      compressed: format(compressedInfo.compressed),
    };
  }, [compressedInfo]);

  const processFile = async (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file');
      return;
    }
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    const previewUrl = URL.createObjectURL(file);
    setSelectedFile(file);
    setPreview(previewUrl);
    setError('');
    setSuccess(false);

    try {
      const compressed = await compressImage(file);
      if (compressedPreview) {
        URL.revokeObjectURL(compressedPreview);
      }
      setCompressedPreview(URL.createObjectURL(compressed));
      setCompressedInfo({ original: file.size, compressed: compressed.size });
      setSelectedFile(compressed);
    } catch (err) {
      console.warn('Image compression failed, falling back to original file:', err);
      setCompressedPreview('');
      setCompressedInfo(null);
      setSelectedFile(file);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      void processFile(file);
    }
    // reset input so same file can be selected again
    e.target.value = '';
  };

  const handleDragOver = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (isDragging) {
      setIsDragging(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const simulateExtraction = () => {
    return {};
  };

  const handleUpload = async () => {
    if (!selectedFile || !user) return;

    setUploading(true);
    setError('');

    try {
      const formData = new FormData();
      const originalName = selectedFile.name;
      const uploadName = originalName.includes('.') ? originalName : `${originalName}.jpg`;
      formData.append('file', selectedFile, uploadName);

      const uploadResponse = await fetch(`${API_BASE_URL}/upload-image`, {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error('Failed to upload image to backend');
      }

      const uploadData = await uploadResponse.json();
      if (uploadData.status !== 'success' || !uploadData.file_url) {
        throw new Error(uploadData.message || 'Unexpected response from upload endpoint');
      }

      const extractedDetails = simulateExtraction();

      const { data: insertedImage, error: insertError } = await supabase
        .from('medical_images')
        .insert({
          user_id: user.id,
          image_url: uploadData.file_url,
          image_name: selectedFile.name,
          extracted_details: extractedDetails,
        })
        .select()
        .single();

      if (insertError || !insertedImage) throw insertError;

      setSuccess(true);
      onImageUploaded(insertedImage as MedicalImage);
      setTimeout(() => {
        if (preview) {
          URL.revokeObjectURL(preview);
        }
        setSelectedFile(null);
        setPreview('');
        setSuccess(false);
      }, 1500);
    } catch (err: any) {
      setError(err.message || 'Failed to upload image');
    } finally {
      setUploading(false);
    }
  };

  const clearSelection = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    if (compressedPreview) {
      URL.revokeObjectURL(compressedPreview);
    }
    setSelectedFile(null);
    setPreview('');
    setCompressedPreview('');
    setCompressedInfo(null);
    setError('');
    setSuccess(false);
    setIsDragging(false);
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="bg-blue-100 p-2 rounded-lg">
          <ImageIcon className="w-6 h-6 text-blue-600" />
        </div>
        <h2 className="text-2xl font-bold text-gray-800">Upload a Medical Image</h2>
      </div>

      {!preview ? (
        <label
          className="block"
          onDragEnter={handleDragOver}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div
            className={`border-2 border-dashed rounded-xl p-12 text-center transition-all cursor-pointer ${
              isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-500 hover:bg-blue-50'
            }`}
          >
            <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-lg font-medium text-gray-700 mb-2">
              Click to upload medical image
            </p>
            <p className="text-sm text-gray-500">
              Supports: X-rays, retina images etc.
            </p>
          </div>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
          />
        </label>
      ) : (
        <div className="space-y-4">
          <div className="relative">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="relative">
                <img
                  src={preview}
                  alt="Original preview"
                  className="w-full h-64 object-contain bg-gray-100 rounded-lg"
                />
                <span className="absolute bottom-2 left-2 bg-white/85 text-xs font-medium px-2 py-1 rounded">
                  Original
                </span>
              </div>
              {compressedPreview && (
                <div className="relative">
                  <img
                    src={compressedPreview}
                    alt="Compressed preview"
                    className="w-full h-64 object-contain bg-gray-100 rounded-lg"
                  />
                  <span className="absolute bottom-2 left-2 bg-white/85 text-xs font-medium px-2 py-1 rounded">
                    Optimized
                  </span>
                </div>
              )}
            </div>

            {!success && (
              <button
                onClick={clearSelection}
                className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 text-white p-2 rounded-lg transition-colors"
                type="button"
              >
                <X className="w-5 h-5" />
              </button>
            )}
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm font-medium text-gray-700 mb-1">Prepared File:</p>
            <p className="text-sm text-gray-600">{selectedFile?.name}</p>
            {humanReadableSize && (
              <p className="text-xs text-gray-500 mt-2">
                Size reduced from {humanReadableSize.original} to {humanReadableSize.compressed}
              </p>
            )}
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
              {error}
            </div>
          )}

          {success && (
            <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-lg text-sm flex items-center gap-2">
              <CheckCircle className="w-5 h-5" />
              <span>Image uploaded and analyzed successfully!</span>
            </div>
          )}

          <button
            onClick={handleUpload}
            disabled={uploading || success}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {uploading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Analyzing Image...</span>
              </>
            ) : success ? (
              <>
                <CheckCircle className="w-5 h-5" />
                <span>Completed</span>
              </>
            ) : (
              'Upload & Analyze'
            )}
          </button>
        </div>
      )}
    </div>
  );
}
