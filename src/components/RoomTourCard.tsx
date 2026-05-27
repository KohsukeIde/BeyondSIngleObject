import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

interface RoomTourCardProps {
  title: string;
  roomType: string;
  location: string;
  videoUrl: string;
  onClick: () => void;
}

const roomTypeLabels: Record<string, string> = {
  bedroom: "Bedroom",
  bathroom: "Bathroom",
  living_room: "Living Room",
};

const roomTypeColors: Record<string, string> = {
  bedroom: "bg-blue-500/10 text-blue-700 dark:text-blue-300",
  bathroom: "bg-purple-500/10 text-purple-700 dark:text-purple-300",
  living_room: "bg-green-500/10 text-green-700 dark:text-green-300",
};

export default function RoomTourCard({
  title,
  roomType,
  location,
  videoUrl,
  onClick,
}: RoomTourCardProps) {
  return (
    <Card
      className="cursor-pointer transition-all hover:scale-[1.02] hover:shadow-lg group overflow-hidden"
      onClick={onClick}
    >
      <div className="relative aspect-video bg-muted overflow-hidden">
        <video
          src={`${videoUrl}#t=0.1`}
          className="w-full h-full object-cover"
          preload="metadata"
          muted
          playsInline
        />
        <div className="absolute inset-0 bg-gradient-to-t from-background/80 via-background/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
          <div className="absolute bottom-0 left-0 right-0 p-4">
            <p className="text-sm font-medium text-foreground">
              Click to view 3D point cloud
            </p>
          </div>
        </div>
        <div className="absolute top-3 right-3">
          <span
            className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-medium ${roomTypeColors[roomType] || "bg-gray-500/10 text-gray-700 dark:text-gray-300"}`}
          >
            {roomTypeLabels[roomType] || roomType}
          </span>
        </div>
      </div>
      <CardHeader className="pb-3">
        <CardTitle className="text-lg">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground">{location}</p>
      </CardContent>
    </Card>
  );
}
