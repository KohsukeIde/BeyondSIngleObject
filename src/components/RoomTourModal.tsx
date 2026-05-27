import { X } from "lucide-react";
import { useEffect } from "react";
import PointCloudViewer from "./PointCloudViewer";
import { Button } from "./ui/button";

interface RoomTourModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  roomType: string;
  location: string;
  videoUrl: string;
  plyUrl: string;
  isDarkMode?: boolean;
}

export default function RoomTourModal({
  isOpen,
  onClose,
  title,
  roomType,
  location,
  videoUrl,
  plyUrl,
  isDarkMode = true,
}: RoomTourModalProps) {
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
      const handleEscape = (e: KeyboardEvent) => {
        if (e.key === "Escape") {
          onClose();
        }
      };
      window.addEventListener("keydown", handleEscape);
      return () => {
        window.removeEventListener("keydown", handleEscape);
      };
    } else {
      document.body.style.overflow = "";
    }
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-background/95 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Modal content */}
      <div className="relative w-full h-full max-w-[95vw] max-h-[95vh] m-4 flex flex-col bg-card border rounded-xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b bg-card/50 backdrop-blur-sm">
          <div className="flex-1 min-w-0">
            <h2 className="text-lg font-semibold truncate">{title}</h2>
            <p className="text-sm text-muted-foreground">
              {location} • {roomType.replace("_", " ")}
            </p>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="ml-4 flex-shrink-0"
          >
            <X className="h-5 w-5" />
            <span className="sr-only">Close</span>
          </Button>
        </div>

        {/* Split view content */}
        <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
          {/* Left panel: Video */}
          <div className="flex-1 flex items-center justify-center bg-black p-4 lg:p-6">
            <div className="w-full h-full flex items-center justify-center">
              <video
                src={videoUrl}
                controls
                autoPlay
                loop
                muted
                playsInline
                className="max-w-full max-h-full rounded-lg shadow-lg"
              >
                <track kind="captions" />
                Your browser does not support the video tag.
              </video>
            </div>
          </div>

          {/* Divider */}
          <div className="h-px lg:h-auto lg:w-px bg-border" />

          {/* Right panel: Point Cloud */}
          <div className="flex-1 flex flex-col bg-muted/30 p-4 lg:p-6">
            <div className="flex-1 relative rounded-lg overflow-hidden bg-background/50 border">
              <PointCloudViewer
                plyUrl={plyUrl}
                isDarkMode={isDarkMode}
                onError={(error) => {
                  console.error("Point cloud error:", error);
                }}
              />
            </div>
            <div className="mt-3 text-center">
              <p className="text-xs text-muted-foreground">
                Drag to rotate • Scroll to zoom • Ctrl+Drag to pan
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
