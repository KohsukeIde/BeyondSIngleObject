# Beyond Single Object - Project Website

This is the GitHub Pages website for the "Beyond Single Object: Learning 3D Relations with Large Language Models" research project.

## About

**Beyond Single Object** introduces Multi-3DLLM, a novel framework for multi-object 3D understanding and comparison. The project includes:

- **MO3D Dataset**: ~70k high-quality QA pairs for multi-object comparison
- **Multi-3DLLM Architecture**: Patch-Interaction Transformer for cross-object geometric reasoning
- **Mini-Apps Benchmarks**: Shape Mating and Change Captioning tasks

Visit the live site: [https://kohsukeide.github.io/BeyondSingleObject/](https://kohsukeide.github.io/BeyondSingleObject/)

## Development

### Prerequisites
- Node.js (v18 or higher)
- Yarn or npm

### Setup
```bash
# Install dependencies
yarn install
# or
npm install
```

### Local Development
```bash
# Start development server
yarn dev
# or
npm run dev
```

The site will be available at `http://localhost:5173`

### Build for Production
```bash
# Build the project
yarn build
# or
npm run build
```

### Deployment
This site is automatically deployed to GitHub Pages when changes are pushed to the `gh-pages` branch.

## Project Structure
- `/src` - React application source code
- `/public` - Static assets (images, videos, etc.)
- `/commands` - Command documentation
- `/environments` - Environment configurations

## Technology Stack
- React 19
- React Router 7
- TypeScript
- Vite 6
- Tailwind CSS 4
- Three.js (for 3D visualizations)

## License
See the main repository for license information.
