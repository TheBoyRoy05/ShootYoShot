import { useState } from "react";
import { toast } from "sonner";
import axios from "axios";
import { sleep } from "../Utils/functions";

interface HTTPProps {
  url: string;
  method: string;
  body?: object;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  handleData?: (data: any) => void;
  handleSuccess?: () => void;
  handleError?: (error: unknown) => void;
  retries?: number;
  retryDelay?: number;
}

const useHTTP = () => {
  const [loading, setLoading] = useState(false);

  const http = async ({
    url,
    method,
    body,
    handleData,
    handleSuccess,
    handleError,
    retries = 2,
    retryDelay = 500,
  }: HTTPProps): Promise<boolean> => {
    setLoading(true);

    let attempt = 0;
    while (attempt <= retries) {
      try {
        const { data } = await axios({
          method,
          headers: { "Content-Type": "application/json" },
          url: `http://localhost:8000${url}`,
          data: body,
        });

        if (data.error) throw new Error(data.error);

        handleData?.(data);
        handleSuccess?.();
        return true;
      } catch (error) {
        attempt++;
        const isLastAttempt = attempt > retries;

        console.error(`Attempt ${attempt} failed`, error);

        if (isLastAttempt) {
          const message = axios.isAxiosError(error)
            ? error.response?.data?.error || error.message
            : "An unexpected error occurred";
          toast.error(message);
          handleError?.(error);
          return false;
        }

        // Wait before next retry
        await sleep(retryDelay);
      } finally {
        if (attempt > retries) setLoading(false);
      }
    }

    setLoading(false);
    return false;
  };

  return { http, loading };
};

export default useHTTP;
