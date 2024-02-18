import React, {useEffect, useContext, useState} from "react"
import {HashLink as Link} from "react-router-hash-link"
import {ThemeContext, EnableDragContext, ExpandDialogFlagContext, HorizontalExpandContext, VerticalExpandContext,
ExpandImageContext, ExpandMaskContext, ImageInputContext} from "../Context"
import functions from "../structures/Functions"
import "./styles/expanddialog.less"
import Draggable from "react-draggable"

const ExpandDialog: React.FunctionComponent = (props) => {
    const {theme, setTheme} = useContext(ThemeContext)
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {expandDialogFlag, setExpandDialogFlag} = useContext(ExpandDialogFlagContext)
    const {horizontalExpand, setHorizontalExpand} = useContext(HorizontalExpandContext)
    const {verticalExpand, setVerticalExpand} = useContext(VerticalExpandContext)
    const {expandImage, setExpandImage} = useContext(ExpandImageContext)
    const {expandMask, setExpandMask} = useContext(ExpandMaskContext)
    const {imageInput, setImageInput} = useContext(ImageInputContext)
    const [lastHorizontal, setLastHorizontal] = useState("0")
    const [lastVertical, setLastVertical] = useState("0")

    useEffect(() => {
        if (expandDialogFlag) {
            setLastHorizontal(horizontalExpand)
            setLastVertical(verticalExpand)
        }
    }, [expandDialogFlag])

    const processExpand = async (image: string, horizontal: number, vertical: number) => {
        const img = document.createElement("img")
        await new Promise<void>((resolve) => {
            img.onload = () => resolve()
            img.src = image
        })
        const normalized = functions.getNormalizedDimensions(img)
        const canvas = document.createElement("canvas")
        canvas.width = normalized.width + Math.floor(horizontal * 2)
        canvas.height = normalized.height + Math.floor(vertical * 2)
        const ctx = canvas.getContext("2d")!
        ctx.fillStyle = "white"
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        ctx.drawImage(img, 0, 0, img.width, img.height, horizontal, vertical, normalized.width, normalized.height)
        const mask = document.createElement("canvas")
        mask.width = normalized.width + Math.floor(horizontal * 2)
        mask.height = normalized.height + Math.floor(vertical * 2)
        const maskCtx = mask.getContext("2d")!
        maskCtx.fillStyle = "white"
        maskCtx.fillRect(0, 0, mask.width, mask.height)
        maskCtx.fillStyle = "black"
        maskCtx.filter = "blur(20px)"
        maskCtx.fillRect(horizontal + 10, vertical + 10, normalized.width - 20, normalized.height - 20)
        const expandedImage = canvas.toDataURL("image/png")
        const expandedMask = mask.toDataURL("image/png")
        setExpandImage(expandedImage)
        setExpandMask(expandedMask)
    }

    const click = (button: "accept" | "reject") => {
        if (button === "accept") {
            const horizontal = Math.round(Number(horizontalExpand))
            const vertical = Math.round(Number(verticalExpand))
            if (Number.isNaN(horizontal) || Number.isNaN(vertical) || (horizontal === 0 && vertical === 0)) {
                setVerticalExpand("0")
                setHorizontalExpand("0")
                setExpandImage("")
                setExpandMask("")
            } else {
                processExpand(imageInput, horizontal, vertical)
            }
        } else {
            setVerticalExpand(lastVertical)
            setHorizontalExpand(lastHorizontal)
        }
        setExpandDialogFlag(false)
    }

    if (expandDialogFlag) {
        return (
            <div className="expand-dialog">
                <Draggable handle=".expand-dialog-title-container">
                <div className="expand-dialog-box" onMouseEnter={() => setEnableDrag(false)} onMouseLeave={() => setEnableDrag(true)}>
                    <div className="expand-container">
                        <div className="expand-dialog-title-container">
                            <span className="expand-dialog-title">Expand Image</span>
                        </div>
                        <div className="expand-dialog-row">
                            <span className="expand-dialog-text">Vertical: </span>
                            <input className="expand-dialog-input" type="number" spellCheck={false} value={verticalExpand} onChange={(event) => setVerticalExpand(event.target.value)}/>
                        </div>
                        <div className="expand-dialog-row">
                            <span className="expand-dialog-text">Horizontal: </span>
                            <input className="expand-dialog-input" type="number" spellCheck={false} value={horizontalExpand} onChange={(event) => setHorizontalExpand(event.target.value)}/>
                        </div>
                        <div className="expand-dialog-row">
                            <button onClick={() => click("reject")} className="expand-button">{"Cancel"}</button>
                            <button onClick={() => click("accept")} className="expand-button">{"Expand"}</button>
                        </div>
                    </div>
                </div>
                </Draggable>
            </div>
        )
    }
    return null
}

export default ExpandDialog