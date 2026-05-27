import { useState } from "react";
import { Box, Image as ImageIcon, ChevronLeft, ChevronRight } from "lucide-react";

import { Tabs, TabsList, TabsTrigger, TabsContent } from "./ui/tabs";
import { Button } from "./ui/button";
import PointCloudViewer from "./PointCloudViewer";
import examplesData from "../data/examples.json";

const BASE = import.meta.env.BASE_URL;

type Example = {
  id: string;
  n: number;
  question: string;
  answer: string;
  reason?: string;
};
type Task = {
  key: string;
  label: string;
  blurb: string;
  objectLabels: string[];
  examples: Example[];
};

function ObjectTile({
  src,
  label,
  show3d,
}: {
  src: string;
  label: string;
  show3d: boolean;
}) {
  return (
    <div className="space-y-1.5">
      <div
        className={
          // Light background in both modes so the rotatable cloud sits on the
          // same light field as the static render — keeps colors consistent
          // when toggling between render and 3D.
          "relative aspect-[10/7] overflow-hidden rounded-lg border " +
          (show3d
            ? "bg-gradient-to-b from-white to-slate-100"
            : "bg-white")
        }
      >
        {show3d ? (
          <PointCloudViewer key={src} plyUrl={`${src}.ply`} />
        ) : (
          <img
            src={`${src}.jpg`}
            alt={label}
            className="h-full w-full object-contain"
            loading="lazy"
          />
        )}
      </div>
      <p className="text-center text-xs text-muted-foreground">{label}</p>
    </div>
  );
}

function TaskGallery({ task }: { task: Task }) {
  const [idx, setIdx] = useState(0);
  const [show3d, setShow3d] = useState(false);
  const ex = task.examples[idx];

  const go = (next: number) => {
    setShow3d(false);
    setIdx((next + task.examples.length) % task.examples.length);
  };

  return (
    <div className="space-y-4 pt-4">
      <p className="text-sm text-muted-foreground">{task.blurb}</p>

      {/* Object renders */}
      <div
        className="grid gap-3"
        style={{ gridTemplateColumns: `repeat(${ex.n}, minmax(0, 1fr))` }}
      >
        {Array.from({ length: ex.n }).map((_, i) => (
          <ObjectTile
            key={i}
            src={`${BASE}examples/${task.key}/${ex.id}/obj${i + 1}`}
            label={task.objectLabels[i] ?? `Object ${i + 1}`}
            show3d={show3d}
          />
        ))}
      </div>

      {/* Q & A */}
      <div className="rounded-xl border bg-muted/30 p-4 text-sm">
        <p>
          <span className="font-semibold text-primary">Q.</span> {ex.question}
        </p>
        <p className="mt-1.5">
          <span className="font-semibold">A.</span> {ex.answer}
        </p>
        {ex.reason && (
          <p className="mt-1.5 text-muted-foreground italic">{ex.reason}</p>
        )}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Button variant="outline" size="icon" onClick={() => go(idx - 1)}>
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="text-sm text-muted-foreground tabular-nums">
            {idx + 1} / {task.examples.length}
          </span>
          <Button variant="outline" size="icon" onClick={() => go(idx + 1)}>
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
        <Button
          variant={show3d ? "default" : "outline"}
          size="sm"
          onClick={() => setShow3d((v) => !v)}
        >
          {show3d ? (
            <>
              <ImageIcon className="mr-2 h-4 w-4" />
              Show renders
            </>
          ) : (
            <>
              <Box className="mr-2 h-4 w-4" />
              View in 3D
            </>
          )}
        </Button>
      </div>
      {show3d && (
        <p className="text-center text-xs text-muted-foreground">
          Drag to rotate · scroll to zoom · ⌘/Ctrl-drag to pan
        </p>
      )}
    </div>
  );
}

export default function InteractiveExamples() {
  const tasks = examplesData.tasks as Task[];
  return (
    <Tabs defaultValue={tasks[0].key} className="w-full">
      <TabsList className="w-full">
        {tasks.map((t) => (
          <TabsTrigger key={t.key} value={t.key}>
            {t.label}
          </TabsTrigger>
        ))}
      </TabsList>
      {tasks.map((t) => (
        <TabsContent key={t.key} value={t.key}>
          <TaskGallery task={t} />
        </TabsContent>
      ))}
    </Tabs>
  );
}
