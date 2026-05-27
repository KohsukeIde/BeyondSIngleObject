import type { MetaDescriptor } from "react-router";

const SITE_URL = "https://kohsukeide.github.io/BeyondSingleObject";
const DEFAULT_IMAGE = `${SITE_URL}/beyond-single-object-og.jpg`;
const DEFAULT_IMAGE_ALT =
  "Beyond Single Object: Multi-3DLLM framework for multi-object 3D understanding and comparison";
const SITE_NAME =
  "Beyond Single Object: Learning 3D Relations with Large Language Models";
const DEFAULT_DESCRIPTION =
  "Multi-3DLLM: A novel framework for multi-object 3D understanding and comparison. Introducing MO3D dataset with 70k examples and Mini-Apps benchmarks for geometric reasoning.";
const DEFAULT_KEYWORDS = [
  "3D large language models",
  "multi-object understanding",
  "point cloud comparison",
  "geometric reasoning",
  "3D vision-language models",
  "shape mating",
  "change captioning",
  "MO3D dataset",
  "Multi-3DLLM",
  "3D-LLM",
  "point cloud understanding",
  "3D object comparison",
  "patch interaction transformer",
];

type SeoConfig = {
  title?: string;
  description?: string;
  path?: string;
  image?: string;
  imageAlt?: string;
  type?: string;
  keywords?: string[];
};

const normalizePath = (path?: string) => {
  if (!path || path === "/") {
    return "";
  }

  return path.startsWith("/") ? path : `/${path}`;
};

export function buildMeta(config: SeoConfig = {}): MetaDescriptor[] {
  const {
    title = SITE_NAME,
    description = DEFAULT_DESCRIPTION,
    path,
    image = DEFAULT_IMAGE,
    imageAlt = DEFAULT_IMAGE_ALT,
    type = "website",
    keywords = [],
  } = config;

  const canonicalPath = normalizePath(path);
  const url = `${SITE_URL}${canonicalPath}`;
  const keywordSet = new Set([
    ...DEFAULT_KEYWORDS,
    ...keywords.filter((keyword) => keyword.trim().length > 0),
  ]);
  const keywordContent = Array.from(keywordSet).join(", ");

  const descriptors: MetaDescriptor[] = [
    { title },
    { name: "description", content: description },
    ...(keywordContent ? [{ name: "keywords", content: keywordContent }] : []),
    { property: "og:type", content: type },
    { property: "og:site_name", content: SITE_NAME },
    { property: "og:title", content: title },
    { property: "og:description", content: description },
    { property: "og:url", content: url },
    { property: "og:image", content: image },
    { property: "og:image:alt", content: imageAlt },
    { name: "twitter:card", content: "summary_large_image" },
    { name: "twitter:title", content: title },
    { name: "twitter:description", content: description },
    { name: "twitter:image", content: image },
    { name: "twitter:image:alt", content: imageAlt },
    { tagName: "link", rel: "canonical", href: url },
  ];

  return descriptors;
}

export function buildScholarlyArticleSchema(config: {
  title: string;
  description: string;
  authors: Array<{ name: string; url?: string }>;
  datePublished?: string;
  keywords?: string[];
  url: string;
  image?: string;
}) {
  return {
    "@context": "https://schema.org",
    "@type": "ScholarlyArticle",
    headline: config.title,
    abstract: config.description,
    author: config.authors.map((author) => ({
      "@type": "Person",
      name: author.name,
      ...(author.url && { url: author.url }),
    })),
    ...(config.datePublished && { datePublished: config.datePublished }),
    keywords: config.keywords?.join(", "),
    url: config.url,
    ...(config.image && { image: config.image }),
  };
}

export const seoDefaults = {
  SITE_NAME,
  SITE_URL,
  DEFAULT_DESCRIPTION,
  DEFAULT_IMAGE,
  DEFAULT_IMAGE_ALT,
  DEFAULT_KEYWORDS,
};
