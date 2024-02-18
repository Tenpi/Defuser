import React, {useContext, useEffect, useRef, useState, useReducer} from "react"
import {useHistory} from "react-router-dom"
import {ImageInputContext, InterrogateTextContext, MaskImageContext, MaskDataContext, ExpandImageContext, ExpandMaskContext,
HorizontalExpandContext, VerticalExpandContext} from "../Context"
import {HashLink as Link} from "react-router-hash-link"
import functions from "../structures/Functions"
import path from "path"
import fileType from "magic-bytes.js"
import "./styles/draganddrop.less"

let showDrag = false
let timeout = null as any

const DragAndDrop: React.FunctionComponent = (props) => {
    const [ignored, forceUpdate] = useReducer(x => x + 1, 0)
    const [visible, setVisible] = useState(false)
    const {imageInput, setImageInput} = useContext(ImageInputContext)
    const {interrogateText, setInterrogateText} = useContext(InterrogateTextContext)
    const {maskImage, setMaskImage} = useContext(MaskImageContext)
    const {maskData, setMaskData} = useContext(MaskDataContext)
    const {expandImage, setExpandImage} = useContext(ExpandImageContext)
    const {expandMask, setExpandMask} = useContext(ExpandMaskContext)
    const {horizontalExpand, setHorizontalExpand} = useContext(HorizontalExpandContext)
    const {verticalExpand, setVerticalExpand} = useContext(VerticalExpandContext)
    const [uploadHover, setUploadHover] = useState(false)
    const history = useHistory()

    const placebo = (event: any) => {
        event.preventDefault()
    }

    const dragOver = (event: any) => {
        event.preventDefault()
        setVisible(true)
    }

    const dragEnd = (event: any) => {
        event.preventDefault()
        clearTimeout(timeout)
        timeout = setTimeout(() => {
            if (!showDrag) setVisible(false) 
        }, 0)
    }

    useEffect(() => {
        window.addEventListener("dragover", placebo)
        window.addEventListener("dragenter", dragOver)
        window.addEventListener("dragleave", dragEnd)
        return () => {
            window.removeEventListener("dragover", placebo)
            window.removeEventListener("dragenter", dragOver)
            window.removeEventListener("dragleave", dragEnd)
        }
    }, [])

    
    useEffect(() => {
        if (!uploadHover) {
            showDrag = false
            setVisible(false)
        }
    }, [uploadHover])

    const dragEnter = (event: React.DragEvent, type: string) => {
        event.preventDefault()
        // window.focus()
        showDrag = true
        setUploadHover(true)
    }

    const dragLeave = (event: React.DragEvent, type: string) => {
        event.preventDefault()
        setUploadHover(false)
    }

    const loadImage = async (file: any) => {
        const fileReader = new FileReader()
        await new Promise<void>((resolve) => {
            fileReader.onloadend = async (f: any) => {
                let bytes = new Uint8Array(f.target.result)
                const result = fileType(bytes)?.[0]
                const jpg = result?.mime === "image/jpeg" ||
                path.extname(file.name).toLowerCase() === ".jpg" ||
                path.extname(file.name).toLowerCase() === ".jpeg"
                const png = result?.mime === "image/png"
                const webp = result?.mime === "image/webp"
                if (jpg) result.typename = "jpg"
                if (jpg || png || webp) {
                    const url = functions.arrayBufferToBase64(bytes.buffer)
                    const link = `${url}#.${result.typename}`
                    removeImage()
                    const metadata = await functions.getImageMetaData(link)
                    if (metadata) setInterrogateText(metadata)
                    setTimeout(() => {
                        setImageInput(link)
                    }, 100)
                }
                resolve()
            }
            fileReader.readAsArrayBuffer(file)
        })
    }

    const removeImage = (event?: any) => {
        setImageInput("")
        setMaskImage("")
        setMaskData("")
        setHorizontalExpand("0")
        setVerticalExpand("0")
        setExpandImage("")
        setExpandMask("")
    }


    const uploadDrop = (event: React.DragEvent) => {
        event.preventDefault()
        setUploadHover(false)
        const files = event.dataTransfer.files 
        if (!files?.[0]) return
        loadImage(files[0])
    }

    return (
        <div className="dragdrop" style={{display: visible ? "flex" : "none"}}>
            <div className="dragdrop-container">
                <div className={`dragdrop-box ${uploadHover ? "dragdrop-hover" : ""}`} onDrop={uploadDrop}
                onDragEnter={(event) => dragEnter(event, "upload")} 
                onDragLeave={(event) => dragLeave(event, "upload")}>
                    Upload
                </div>
            </div>
        </div>
    )
}

export default DragAndDrop