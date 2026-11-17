import { navigate } from "../navigation";

const router = {
  push(path: string) {
    const clean = path.replace(/^#/, "");
    const [destination, query = ""] = clean.split("?");
    const params = new URLSearchParams(query);
    params.set("back", window.location.hash.slice(1));
    navigate(`${destination}?${params}`);
  },
};
export const useRouter = () => router;
