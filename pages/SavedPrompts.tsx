import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {HashLink as Link} from "react-router-hash-link"
import favicon from "../assets/icons/favicon.png"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, 
SavedPromptsContext, SavedPromptsNovelAIContext, SavedPromptsHolaraAIContext, GeneratorContext} from "../Context"
import functions from "../structures/Functions"
import CheckpointBar from "../components/CheckpointBar"
import "./styles/generate.less"
import axios from "axios"
import imagesMeta from "images-meta"
import path from "path"

let timeout = null as any

const SavedPrompts: React.FunctionComponent = (props) => {
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {savedPrompts, setSavedPrompts} = useContext(SavedPromptsContext)
    const {savedPromptsNovelAI, setSavedPromptsNovelAI} = useContext(SavedPromptsNovelAIContext)
    const {savedPromptsHolaraAI, setSavedPromptsHolaraAI} = useContext(SavedPromptsHolaraAIContext)
    const {generator, setGenerator} = useContext(GeneratorContext)
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    useEffect(() => {
        const savedPrompts = localStorage.getItem("savedPrompts")
        if (savedPrompts) setSavedPrompts(JSON.parse(savedPrompts))
        const savedPromptsNovelAI = localStorage.getItem("savedPrompts-novel-ai")
        if (savedPromptsNovelAI) setSavedPromptsNovelAI(JSON.parse(savedPromptsNovelAI))
        const savedPromptsHolaraAI = localStorage.getItem("savedPrompts-holara-ai")
        if (savedPromptsHolaraAI) setSavedPromptsHolaraAI(JSON.parse(savedPromptsHolaraAI))
    }, [])

    useEffect(() => {
        localStorage.setItem("savedPrompts", JSON.stringify(savedPrompts))
        localStorage.setItem("savedPrompts-novel-ai", JSON.stringify(savedPromptsNovelAI))
        localStorage.setItem("savedPrompts-holara-ai", JSON.stringify(savedPromptsHolaraAI))
    }, [savedPrompts, savedPromptsNovelAI, savedPromptsHolaraAI])

    const changeSavedPrompts = (value: string) => {
        if (timeout) clearTimeout(timeout)
        const prompts = value.split("\n")
        if (generator === "novel ai") {
            setSavedPromptsNovelAI(prompts)
        } else if (generator === "holara ai") {
            setSavedPromptsHolaraAI(prompts)
        } else {
            setSavedPrompts(prompts)
        }
        timeout = setTimeout(() => {
            axios.post("/save-prompts", {prompts, generator})
        }, 1000)
    }

    const getSavedPrompts = () => {
        if (generator === "novel ai") return savedPromptsNovelAI.join("\n")
        if (generator === "holara ai") return savedPromptsHolaraAI.join("\n")
        return savedPrompts.join("\n")
    }

    const getSaveKey = () => {
        if (generator === "novel ai") return "saved-novel-ai"
        if (generator === "holara ai") return "saved-holara-ai"
        return "saved"
    }

    const getPrompt = async (img: string) => {
        let inMime = "image/jpeg"
        if (path.extname(img) === ".png") inMime = "image/png"
        if (path.extname(img) === ".webp") inMime = "image/webp"
        const arrayBuffer = await fetch((img)).then((r) => r.arrayBuffer())
        if (inMime === "image/png") {
            const meta = imagesMeta.readMeta(Buffer.from(arrayBuffer), inMime)
            const prompt = meta.find((m: any) => m.name?.toLowerCase() === "prompt")?.value || ""
            return prompt
        } else if (inMime === "image/jpeg") {
            let str = ""
            const meta = imagesMeta.readMeta(Buffer.from(arrayBuffer), inMime)
            for (let i = 0; i < meta.length; i++) {
                if (meta[i].name?.toLowerCase() === "usercomment") {
                    str +=  meta[i].value.replaceAll("UNICODE", "").replaceAll(/\u0000/g, "")
                }
            }
            const {prompt} = functions.extractMetaValues(str)
            return prompt
        } else {
            const form = new FormData()
            const blob = new Blob([new Uint8Array(arrayBuffer)])
            const file = new File([blob],img)
            form.append("image", file)
            const exif = await axios.post("get-exif", form).then((r) => r.data)
            const str = exif.replaceAll("UNICODE", "").replaceAll(/\u0000/g, "")
            const {prompt} = functions.extractMetaValues(str)
            return prompt
        }
    }

    const importSaved = async () => {
        let saved = localStorage.getItem(getSaveKey()) || "[]" as any
        saved = JSON.parse(saved)
        let prompts = [] as any
        for (let i = 0; i < saved.length; i++) {
            const prompt = await getPrompt(saved[i])
            prompts.push(prompt)
        }
        changeSavedPrompts(prompts.join("\n"))
    }

    return (
        <div className="generate" onMouseEnter={() => setEnableDrag(false)}>
            <CheckpointBar/>
            <div className="settings-bar">
                <div className="settings-bar-row">
                    <span className="settings-bar-text">Saved Prompts:</span>
                    <textarea className="settings-bar-textarea" style={{minHeight: "75vh", width: "70vw"}} spellCheck={false} value={getSavedPrompts()} onChange={(event) => changeSavedPrompts(event.target.value)}></textarea>
                </div>
                <div className="settings-bar-row">
                    <button className="settings-bar-button" onClick={() => importSaved()}>Import From Saved</button>
                </div>
            </div>
        </div>
    )
}

export default SavedPrompts