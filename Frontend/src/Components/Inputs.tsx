import type { FormInputs } from "../Utils/types"

interface InputsProps {
  inputs: FormInputs;
  setInputs: React.Dispatch<React.SetStateAction<FormInputs>>;
}

const Inputs = ({ inputs, setInputs }: InputsProps) => {
  return (
    <div
      className="grid  gap-y-2 gap-x-8
                grid-cols-3
                justify-center"
    >
      {/* ─────────── Row 1 – labels ─────────── */}
      <label htmlFor="gender" className="font-semibold text-center">
        Gender
      </label>

      <label htmlFor="height" className="font-semibold text-center">
        Height&nbsp;(in)
      </label>

      <label htmlFor="weight" className="font-semibold text-center">
        Weight&nbsp;(lbs)
      </label>

      {/* ─────────── Row 2 – inputs ─────────── */}
      <select
        id="gender"
        value={inputs.gender}
        onChange={(e) => setInputs({ ...inputs, gender: e.target.value as "Male" | "Female" })}
        className="select select-bordered w-full text-center"
      >
        <option value="" disabled hidden>
          Select...
        </option>
        <option value="Male">Male</option>
        <option value="Female">Female</option>
      </select>

      <input
        id="height"
        type="number"
        min={48}
        max={96}
        step={1}
        value={inputs.height ?? ""}
        onChange={(e) => setInputs({ ...inputs, height: +e.target.value })}
        className="input input-bordered w-full text-center"
      />
      <input
        id="weight"
        type="number"
        value={inputs.weight ?? ""}
        onChange={(e) => setInputs({ ...inputs, weight: +e.target.value })}
        className="input input-bordered w-full text-center"
      />
    </div>
  );
};

export default Inputs;
