import { useEffect, useState } from 'react';
import { supabase, MedicalImage } from '../lib/supabase';
import { useAuth } from '../contexts/AuthContext';
import { FileImage, Calendar, Activity, Trash2 } from 'lucide-react';

interface ImageListProps {
  refresh: number;
  selectedImage: MedicalImage | null;
  onSelect: (image: MedicalImage | null) => void;
}

export default function ImageList({ refresh, selectedImage, onSelect }: ImageListProps) {
  const [images, setImages] = useState<MedicalImage[]>([]);
  const [loading, setLoading] = useState(true);
  const { user } = useAuth();

  const hasAnalysisDetails = (details: MedicalImage['extracted_details']) => {
    if (!details) {
      return false;
    }

    const bodyPart = (details as any).bodyPart;
    const findings = (details as any).findings;
    const confidence = (details as any).confidence;

    const hasBodyPart = Boolean(bodyPart);
    const hasFindings = Array.isArray(findings) && findings.length > 0;
    const hasConfidence = confidence !== undefined && confidence !== null;

    return hasBodyPart || hasFindings || hasConfidence;
  };

  useEffect(() => {
    loadImages();
  }, [refresh, user]);

  const loadImages = async () => {
    if (!user) return;

    setLoading(true);
    const { data, error } = await supabase
      .from('medical_images')
      .select('*')
      .eq('user_id', user.id)
      .order('created_at', { ascending: false });

    if (!error && data) {
      setImages(data);
      if (
        selectedImage &&
        !data.some((img) => img.id === selectedImage.id)
      ) {
        onSelect(null);
      }
    }
    setLoading(false);
  };

  const deleteImage = async (id: string) => {
    const { error } = await supabase.from('medical_images').delete().eq('id', id);
    if (!error) {
      setImages(images.filter((img) => img.id !== id));
      if (selectedImage?.id === id) {
        onSelect(null);
      }
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-gray-200 rounded w-1/3"></div>
          <div className="h-32 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="bg-cyan-100 p-2 rounded-lg">
          <FileImage className="w-6 h-6 text-cyan-600" />
        </div>
        <h2 className="text-2xl font-bold text-gray-800">Medical Images</h2>
      </div>

      {images.length === 0 ? (
        <div className="text-center py-12">
          <FileImage className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <p className="text-gray-500">No images uploaded yet</p>
        </div>
      ) : (
        <div className="space-y-4">
          {images.map((image) => {
            const createdAt = new Date(image.created_at);
            const formattedDate = createdAt.toLocaleDateString();
            const formattedTime = createdAt.toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit',
            });

            return (
              <div
                key={image.id}
                className={`border rounded-lg p-4 cursor-pointer transition-all ${
                  selectedImage?.id === image.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-blue-300 hover:bg-gray-50'
                }`}
                onClick={() => onSelect(image)}
              >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <h3 className="font-semibold text-gray-800 mb-2">
                    {image.image_name}
                  </h3>
                  <div className="flex items-center gap-4 text-sm text-gray-600">
                    <div className="flex items-center gap-1">
                      <Calendar className="w-4 h-4" />
                      <span>
                        {formattedDate} at {formattedTime}
                      </span>
                    </div>
                    {image.extracted_details?.type && (
                      <div className="flex items-center gap-1">
                        <Activity className="w-4 h-4" />
                        <span>{image.extracted_details.type}</span>
                      </div>
                    )}
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteImage(image.id);
                  }}
                  className="text-red-500 hover:text-red-700 p-2 hover:bg-red-50 rounded-lg transition-colors"
                >
                  <Trash2 className="w-5 h-5" />
                </button>
              </div>

                {selectedImage?.id === image.id &&
                  hasAnalysisDetails(image.extracted_details) && (
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <h4 className="font-semibold text-gray-700 mb-2">
                      Analysis Details:
                    </h4>
                    <div className="space-y-2 text-sm">
                      {image.extracted_details.bodyPart && (
                        <p>
                          <span className="font-medium">Body Part:</span>{' '}
                          {image.extracted_details.bodyPart}
                        </p>
                      )}
                      {image.extracted_details.findings && (
                        <div>
                          <span className="font-medium">Findings:</span>
                          <ul className="list-disc list-inside ml-2 mt-1">
                            {image.extracted_details.findings.map(
                              (finding: string, idx: number) => (
                                <li key={idx} className="text-gray-600">
                                  {finding}
                                </li>
                              )
                            )}
                          </ul>
                        </div>
                      )}
                      {image.extracted_details.confidence && (
                        <p>
                          <span className="font-medium">Confidence:</span>{' '}
                          {(image.extracted_details.confidence * 100).toFixed(1)}%
                        </p>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
